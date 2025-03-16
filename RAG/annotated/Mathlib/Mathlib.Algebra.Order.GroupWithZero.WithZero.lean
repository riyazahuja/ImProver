instance {α : Type*} [Mul α] [Preorder α] [MulLeftStrictMono α] :
    PosMulStrictMono (WithZero α) where
  elim := @fun
    | ⟨(x : α), hx⟩, 0, (b : α), _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          b : α
          x✝ : LT.lt 0 ↑b
          ⊢ LT.lt (HMul.hMul (↑⟨↑x, hx⟩) 0) (HMul.hMul ↑⟨↑x, hx⟩ ↑b)
        -/
        simpa only [mul_zero] using WithZero.zero_lt_coe _
        /-
          🎉 no goals
        -/
    | ⟨(x : α), hx⟩, (a : α), (b : α), h => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt ↑a ↑b
          ⊢ LT.lt (HMul.hMul ↑⟨↑x, hx⟩ ↑a) (HMul.hMul ↑⟨↑x, hx⟩ ↑b)
        -/
        dsimp only
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt ↑a ↑b
          ⊢ LT.lt (HMul.hMul ↑x ↑a) (HMul.hMul ↑x ↑b)
        -/
        norm_cast at h ⊢
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt a b
          ⊢ LT.lt (HMul.hMul x a) (HMul.hMul x b)
        -/
        exact mul_lt_mul_left' h x
        /-
          🎉 no goals
        -/


open Function in
instance {α : Type*} [Mul α] [Preorder α] [MulRightStrictMono α] :
    MulPosStrictMono (WithZero α) where
  elim := @fun
    | ⟨(x : α), hx⟩, 0, (b : α), _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          b : α
          x✝ : LT.lt 0 ↑b
          ⊢ LT.lt (HMul.hMul 0 ↑⟨↑x, hx⟩) (HMul.hMul ↑b ↑⟨↑x, hx⟩)
        -/
        simpa only [mul_zero] using WithZero.zero_lt_coe _
        /-
          🎉 no goals
        -/
    | ⟨(x : α), hx⟩, (a : α), (b : α), h => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt ↑a ↑b
          ⊢ LT.lt (HMul.hMul ↑a ↑⟨↑x, hx⟩) (HMul.hMul ↑b ↑⟨↑x, hx⟩)
        -/
        dsimp only
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt ↑a ↑b
          ⊢ LT.lt (HMul.hMul ↑a ↑x) (HMul.hMul ↑b ↑x)
        -/
        norm_cast at h ⊢
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightStrictMono α
          x : α
          hx : LT.lt 0 ↑x
          a b : α
          h : LT.lt a b
          ⊢ LT.lt (HMul.hMul a x) (HMul.hMul b x)
        -/
        exact mul_lt_mul_right' h x
        /-
          🎉 no goals
        -/


instance {α : Type*} [Mul α] [Preorder α] [MulLeftMono α] :
    PosMulMono (WithZero α) where
  elim := @fun
    | ⟨0, _⟩, a, b, _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftMono α
          property✝ : LE.le 0 0
          a b : WithZero α
          x✝ : LE.le a b
          ⊢ LE.le (HMul.hMul (↑⟨0, property✝⟩) a) (HMul.hMul (↑⟨0, property✝⟩) b)
        -/
        simp only [zero_mul, le_refl]
        /-
          🎉 no goals
        -/
    | ⟨(x : α), _⟩, 0, _, _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftMono α
          x : α
          property✝ : LE.le 0 ↑x
          x✝¹ : WithZero α
          x✝ : LE.le 0 x✝¹
          ⊢ LE.le (HMul.hMul (↑⟨↑x, property✝⟩) 0) (HMul.hMul (↑⟨↑x, property✝⟩) x✝¹)
        -/
        simp only [mul_zero, WithZero.zero_le]
        /-
          🎉 no goals
        -/
    | ⟨(x : α), _⟩, (a : α), 0, h =>
        (lt_irrefl 0 (lt_of_lt_of_le (WithZero.zero_lt_coe a) h)).elim
    | ⟨(x : α), hx⟩, (a : α), (b : α), h => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le ↑a ↑b
          ⊢ LE.le (HMul.hMul ↑⟨↑x, hx⟩ ↑a) (HMul.hMul ↑⟨↑x, hx⟩ ↑b)
        -/
        dsimp only
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le ↑a ↑b
          ⊢ LE.le (HMul.hMul ↑x ↑a) (HMul.hMul ↑x ↑b)
        -/
        norm_cast at h ⊢
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulLeftMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le a b
          ⊢ LE.le (HMul.hMul x a) (HMul.hMul x b)
        -/
        exact mul_le_mul_left' h x
        /-
          🎉 no goals
        -/

-- This makes `lt_mul_of_le_of_one_lt'` work on `ℤₘ₀`

open Function in
instance {α : Type*} [Mul α] [Preorder α] [MulRightMono α] :
    MulPosMono (WithZero α) where
  elim := @fun
    | ⟨0, _⟩, a, b, _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightMono α
          property✝ : LE.le 0 0
          a b : WithZero α
          x✝ : LE.le a b
          ⊢ LE.le (HMul.hMul a ↑⟨0, property✝⟩) (HMul.hMul b ↑⟨0, property✝⟩)
        -/
        simp only [mul_zero, le_refl]
        /-
          🎉 no goals
        -/
    | ⟨(x : α), _⟩, 0, _, _ => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightMono α
          x : α
          property✝ : LE.le 0 ↑x
          x✝¹ : WithZero α
          x✝ : LE.le 0 x✝¹
          ⊢ LE.le (HMul.hMul 0 ↑⟨↑x, property✝⟩) (HMul.hMul x✝¹ ↑⟨↑x, property✝⟩)
        -/
        simp only [zero_mul, WithZero.zero_le]
        /-
          🎉 no goals
        -/
    | ⟨(x : α), _⟩, (a : α), 0, h =>
        (lt_irrefl 0 (lt_of_lt_of_le (WithZero.zero_lt_coe a) h)).elim
    | ⟨(x : α), hx⟩, (a : α), (b : α), h => by
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le ↑a ↑b
          ⊢ LE.le (HMul.hMul ↑a ↑⟨↑x, hx⟩) (HMul.hMul ↑b ↑⟨↑x, hx⟩)
        -/
        dsimp only
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le ↑a ↑b
          ⊢ LE.le (HMul.hMul ↑a ↑x) (HMul.hMul ↑b ↑x)
        -/
        norm_cast at h ⊢
        /-
          α : Type u_1
          inst✝² : Mul α
          inst✝¹ : Preorder α
          inst✝ : MulRightMono α
          x : α
          hx : LE.le 0 ↑x
          a b : α
          h : LE.le a b
          ⊢ LE.le (HMul.hMul a x) (HMul.hMul b x)
        -/
        exact mul_le_mul_right' h x
        /-
          🎉 no goals
        -/

