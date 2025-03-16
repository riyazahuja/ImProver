instance mulZeroClass : MulZeroClass (∀ i, α i) where
                 /-
                   ι : Type u_1
                   α : ι → Type u_2
                   inst✝¹ : (i : ι) → MulZeroClass (α i)
                   inst✝ : DecidableEq ι
                   i : ι
                   f : (i : ι) → α i
                   ⊢ ∀ (a : (i : ι) → α i), Eq (HMul.hMul 0 a) 0
                 -/
  zero_mul := by intros; ext; exact zero_mul _
                              /-
                                🎉 no goals
                              -/
                 /-
                   ι : Type u_1
                   α : ι → Type u_2
                   inst✝¹ : (i : ι) → MulZeroClass (α i)
                   inst✝ : DecidableEq ι
                   i : ι
                   f : (i : ι) → α i
                   ⊢ ∀ (a : (i : ι) → α i), Eq (HMul.hMul a 0) 0
                 -/
  mul_zero := by intros; ext; exact mul_zero _
                              /-
                                🎉 no goals
                              -/


/-- The multiplicative homomorphism including a single `MulZeroClass`
into a dependent family of `MulZeroClass`es, as functions supported at a point.

This is the `MulHom` version of `Pi.single`. -/
@[simps]
def _root_.MulHom.single (i : ι) : α i →ₙ* ∀ i, α i where
  toFun := Pi.single i
  map_mul' := Pi.single_op₂ (fun _ ↦ (· * ·)) (fun _ ↦ zero_mul _) _


lemma single_mul (i : ι) (x y : α i) : single i (x * y) = single i x * single i y :=
  (MulHom.single _).map_mul _ _


lemma single_mul_left_apply (i j : ι) (a : α i) (f : ∀ i, α i) :
    single i (a * f i) j = single i a j * f j :=
  (apply_single (fun i ↦ (· * f i)) (fun _ ↦ zero_mul _) _ _ _).symm


lemma single_mul_right_apply (i j : ι) (f : ∀ i, α i) (a : α i) :
    single i (f i * a) j = f j * single i a j :=
  (apply_single (f · * ·) (fun _ ↦ mul_zero _) _ _ _).symm


lemma single_mul_left (a : α i) : single i (a * f i) = single i a * f :=
  funext fun _ ↦ single_mul_left_apply _ _ _ _


lemma single_mul_right (a : α i) : single i (f i * a) = f * single i a :=
  funext fun _ ↦ single_mul_right_apply _ _ _ _


instance mulZeroOneClass [∀ i, MulZeroOneClass (α i)] : MulZeroOneClass (∀ i, α i) where
  __ := mulZeroClass
  __ := mulOneClass


instance monoidWithZero [∀ i, MonoidWithZero (α i)] : MonoidWithZero (∀ i, α i) where
  __ := monoid
  __ := mulZeroClass


instance commMonoidWithZero [∀ i, CommMonoidWithZero (α i)] : CommMonoidWithZero (∀ i, α i) where
  __ := monoidWithZero
  __ := commMonoid


instance semigroupWithZero [∀ i, SemigroupWithZero (α i)] : SemigroupWithZero (∀ i, α i) where
  __ := semigroup
  __ := mulZeroClass


