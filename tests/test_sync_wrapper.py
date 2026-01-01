"""Tests for the sync_wrapper decorator.

Tests cover:
1. Basic sync-to-async delegation
2. Auto-detection of async method name (insert -> ainsert)
3. Explicit async method name specification
4. Argument and keyword argument passing
5. Return value propagation
6. Error propagation from async method
7. Introspection attributes (_async_target)
"""

import asyncio

import pytest

from lightrag.utils import sync_wrapper


class TestSyncWrapperBasic:
    """Basic functionality tests for sync_wrapper."""

    def test_auto_detect_async_method_name(self):
        """sync_wrapper() should auto-detect 'amethod' from 'method'."""

        class MyClass:
            async def aprocess(self) -> str:
                return 'async result'

            @sync_wrapper()
            def process(self) -> str:
                pass  # Body replaced by decorator

        obj = MyClass()
        result = obj.process()
        assert result == 'async result'

    def test_explicit_async_method_name(self):
        """sync_wrapper('custom_async') should use explicit target."""

        class MyClass:
            async def custom_async(self) -> str:
                return 'custom result'

            @sync_wrapper('custom_async')
            def my_sync(self) -> str:
                pass

        obj = MyClass()
        result = obj.my_sync()
        assert result == 'custom result'

    def test_passes_positional_arguments(self):
        """Positional arguments should be passed to async method."""

        class MyClass:
            async def aadd(self, a: int, b: int) -> int:
                return a + b

            @sync_wrapper()
            def add(self, a: int, b: int) -> int:
                pass

        obj = MyClass()
        result = obj.add(3, 5)
        assert result == 8

    def test_passes_keyword_arguments(self):
        """Keyword arguments should be passed to async method."""

        class MyClass:
            async def agreeting(self, name: str, greeting: str = 'Hello') -> str:
                return f'{greeting}, {name}!'

            @sync_wrapper()
            def greeting(self, name: str, greeting: str = 'Hello') -> str:
                pass

        obj = MyClass()
        assert obj.greeting('Alice') == 'Hello, Alice!'
        assert obj.greeting('Bob', greeting='Hi') == 'Hi, Bob!'
        assert obj.greeting(name='Charlie', greeting='Hey') == 'Hey, Charlie!'

    def test_passes_mixed_arguments(self):
        """Mixed positional and keyword args should work."""

        class MyClass:
            async def acompute(self, x: int, y: int, *, multiplier: int = 1) -> int:
                return (x + y) * multiplier

            @sync_wrapper()
            def compute(self, x: int, y: int, *, multiplier: int = 1) -> int:
                pass

        obj = MyClass()
        assert obj.compute(2, 3) == 5
        assert obj.compute(2, 3, multiplier=2) == 10

    def test_returns_none_correctly(self):
        """Methods returning None should work."""

        class MyClass:
            def __init__(self):
                self.called = False

            async def aset_flag(self) -> None:
                self.called = True

            @sync_wrapper()
            def set_flag(self) -> None:
                pass

        obj = MyClass()
        result = obj.set_flag()
        assert result is None
        assert obj.called is True


class TestSyncWrapperReturnTypes:
    """Tests for various return types."""

    def test_returns_list(self):
        """List return type should work."""

        class MyClass:
            async def aget_items(self) -> list[str]:
                return ['a', 'b', 'c']

            @sync_wrapper()
            def get_items(self) -> list[str]:
                pass

        obj = MyClass()
        assert obj.get_items() == ['a', 'b', 'c']

    def test_returns_dict(self):
        """Dict return type should work."""

        class MyClass:
            async def aget_data(self) -> dict[str, int]:
                return {'x': 1, 'y': 2}

            @sync_wrapper()
            def get_data(self) -> dict[str, int]:
                pass

        obj = MyClass()
        assert obj.get_data() == {'x': 1, 'y': 2}

    def test_returns_tuple(self):
        """Tuple return type should work."""

        class MyClass:
            async def aget_pair(self) -> tuple[int, str]:
                return (42, 'answer')

            @sync_wrapper()
            def get_pair(self) -> tuple[int, str]:
                pass

        obj = MyClass()
        assert obj.get_pair() == (42, 'answer')

    def test_returns_complex_object(self):
        """Complex object return type should work."""

        class Result:
            def __init__(self, value: int):
                self.value = value

        class MyClass:
            async def aget_result(self) -> Result:
                return Result(100)

            @sync_wrapper()
            def get_result(self) -> Result:
                pass

        obj = MyClass()
        result = obj.get_result()
        assert isinstance(result, Result)
        assert result.value == 100


class TestSyncWrapperErrorHandling:
    """Tests for error propagation."""

    def test_propagates_value_error(self):
        """ValueError from async method should propagate."""

        class MyClass:
            async def avalidate(self, value: int) -> int:
                if value < 0:
                    raise ValueError('Value must be non-negative')
                return value

            @sync_wrapper()
            def validate(self, value: int) -> int:
                pass

        obj = MyClass()
        assert obj.validate(5) == 5

        with pytest.raises(ValueError) as excinfo:
            obj.validate(-1)
        assert 'non-negative' in str(excinfo.value)

    def test_propagates_custom_exception(self):
        """Custom exceptions should propagate."""

        class CustomError(Exception):
            pass

        class MyClass:
            async def afail(self) -> None:
                raise CustomError('Custom failure')

            @sync_wrapper()
            def fail(self) -> None:
                pass

        obj = MyClass()
        with pytest.raises(CustomError) as excinfo:
            obj.fail()
        assert 'Custom failure' in str(excinfo.value)

    def test_propagates_attribute_error_for_missing_async(self):
        """Missing async method should raise AttributeError."""

        class MyClass:
            @sync_wrapper()
            def missing_async(self) -> None:
                pass

        obj = MyClass()
        with pytest.raises(AttributeError) as excinfo:
            obj.missing_async()
        assert 'amissing_async' in str(excinfo.value)


class TestSyncWrapperIntrospection:
    """Tests for introspection and metadata."""

    def test_async_target_attribute(self):
        """Decorated method should have _async_target attribute."""

        class MyClass:
            async def aprocess(self) -> str:
                return 'result'

            @sync_wrapper()
            def process(self) -> str:
                pass

        assert hasattr(MyClass.process, '_async_target')
        assert MyClass.process._async_target == 'aprocess'

    def test_explicit_async_target_attribute(self):
        """Explicit target should be stored in _async_target."""

        class MyClass:
            async def custom(self) -> str:
                return 'result'

            @sync_wrapper('custom')
            def my_method(self) -> str:
                pass

        assert MyClass.my_method._async_target == 'custom'

    def test_preserves_function_name(self):
        """Decorated method should preserve __name__."""

        class MyClass:
            async def aprocess(self) -> str:
                return 'result'

            @sync_wrapper()
            def process(self) -> str:
                """Process synchronously."""
                pass

        assert MyClass.process.__name__ == 'process'

    def test_preserves_docstring(self):
        """Decorated method should preserve docstring if provided."""

        class MyClass:
            async def aprocess(self) -> str:
                """Async processing."""
                return 'result'

            @sync_wrapper()
            def process(self) -> str:
                """Sync processing."""
                pass

        assert MyClass.process.__doc__ == 'Sync processing.'


class TestSyncWrapperAsync:
    """Tests ensuring async behavior is correct."""

    def test_awaits_coroutine(self):
        """Decorator should properly await the async coroutine."""

        class MyClass:
            def __init__(self):
                self.steps = []

            async def asequence(self) -> list[str]:
                self.steps.append('start')
                await asyncio.sleep(0)  # Yield control
                self.steps.append('middle')
                await asyncio.sleep(0)
                self.steps.append('end')
                return self.steps

            @sync_wrapper()
            def sequence(self) -> list[str]:
                pass

        obj = MyClass()
        result = obj.sequence()
        assert result == ['start', 'middle', 'end']

    def test_works_with_nested_async_calls(self):
        """Should work when async method calls other async methods."""

        class MyClass:
            async def _helper(self, x: int) -> int:
                return x * 2

            async def acomplex(self, x: int) -> int:
                a = await self._helper(x)
                b = await self._helper(a)
                return b

            @sync_wrapper()
            def complex(self, x: int) -> int:
                pass

        obj = MyClass()
        assert obj.complex(5) == 20  # 5 * 2 * 2


class TestSyncWrapperMultipleMethods:
    """Tests with multiple decorated methods on same class."""

    def test_multiple_sync_wrappers(self):
        """Multiple methods can use sync_wrapper independently."""

        class MyClass:
            async def ainsert(self, data: str) -> bool:
                return True

            async def aquery(self, query: str) -> str:
                return f'result: {query}'

            async def adelete(self, id: int) -> None:
                pass

            @sync_wrapper()
            def insert(self, data: str) -> bool:
                pass

            @sync_wrapper()
            def query(self, query: str) -> str:
                pass

            @sync_wrapper()
            def delete(self, id: int) -> None:
                pass

        obj = MyClass()
        assert obj.insert('test') is True
        assert obj.query('foo') == 'result: foo'
        assert obj.delete(1) is None


class TestSyncWrapperEdgeCases:
    """Edge case tests."""

    def test_empty_args(self):
        """Method with no args should work."""

        class MyClass:
            async def anoop(self) -> str:
                return 'done'

            @sync_wrapper()
            def noop(self) -> str:
                pass

        obj = MyClass()
        assert obj.noop() == 'done'

    def test_many_args(self):
        """Method with many args should work."""

        class MyClass:
            async def amany(self, a, b, c, d, e, f, g, h) -> int:
                return a + b + c + d + e + f + g + h

            @sync_wrapper()
            def many(self, a, b, c, d, e, f, g, h) -> int:
                pass

        obj = MyClass()
        assert obj.many(1, 2, 3, 4, 5, 6, 7, 8) == 36

    def test_star_args(self):
        """Method with *args should work."""

        class MyClass:
            async def asum(self, *values: int) -> int:
                return sum(values)

            @sync_wrapper()
            def sum(self, *values: int) -> int:
                pass

        obj = MyClass()
        assert obj.sum(1, 2, 3, 4, 5) == 15

    def test_star_kwargs(self):
        """Method with **kwargs should work."""

        class MyClass:
            async def amerge(self, **data: str) -> dict[str, str]:
                return data

            @sync_wrapper()
            def merge(self, **data: str) -> dict[str, str]:
                pass

        obj = MyClass()
        assert obj.merge(a='1', b='2') == {'a': '1', 'b': '2'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
