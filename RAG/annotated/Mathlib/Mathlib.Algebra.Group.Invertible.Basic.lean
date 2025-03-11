/-- An `Invertible` element is a unit. -/
@[simps]
def unitOfInvertible [Monoid α] (a : α) [Invertible a] : αˣ where
  val := a
  inv := ⅟ a
                /-
                  α : Type u
                  inst✝¹ : Monoid α
                  a : α
                  inst✝ : Invertible a
                  ⊢ Eq (HMul.hMul a (Invertible.invOf a)) 1
                -/
  val_inv := by simp
                /-
                  🎉 no goals
                -/
                /-
                  α : Type u
                  inst✝¹ : Monoid α
                  a : α
                  inst✝ : Invertible a
                  ⊢ Eq (HMul.hMul (Invertible.invOf a) a) 1
                -/
  inv_val := by simp
                /-
                  🎉 no goals
                -/


theorem isUnit_of_invertible [Monoid α] (a : α) [Invertible a] : IsUnit a :=
  ⟨unitOfInvertible a, rfl⟩


/-- Units are invertible in their associated monoid. -/
def Units.invertible [Monoid α] (u : αˣ) :
    Invertible (u : α) where
  invOf := ↑u⁻¹
  invOf_mul_self := u.inv_mul
  mul_invOf_self := u.mul_inv


@[simp]
theorem invOf_units [Monoid α] (u : αˣ) [Invertible (u : α)] : ⅟ (u : α) = ↑u⁻¹ :=
  invOf_eq_right_inv u.mul_inv


theorem IsUnit.nonempty_invertible [Monoid α] {a : α} (h : IsUnit a) : Nonempty (Invertible a) :=
  let ⟨x, hx⟩ := h
  ⟨x.invertible.copy _ hx.symm⟩


/-- Convert `IsUnit` to `Invertible` using `Classical.choice`.

Prefer `casesI h.nonempty_invertible` over `letI := h.invertible` if you want to avoid choice. -/
noncomputable def IsUnit.invertible [Monoid α] {a : α} (h : IsUnit a) : Invertible a :=
  Classical.choice h.nonempty_invertible


@[simp]
theorem nonempty_invertible_iff_isUnit [Monoid α] (a : α) : Nonempty (Invertible a) ↔ IsUnit a :=
  ⟨Nonempty.rec <| @isUnit_of_invertible _ _ _, IsUnit.nonempty_invertible⟩


theorem Commute.invOf_right [Monoid α] {a b : α} [Invertible b] (h : Commute a b) :
    Commute a (⅟ b) :=
  calc
                                        /-
                                          α : Type u
                                          inst✝¹ : Monoid α
                                          a b : α
                                          inst✝ : Invertible b
                                          h : Commute a b
                                          ⊢ Eq (HMul.hMul a (Invertible.invOf b)) (HMul.hMul (Invertible.invOf b) (HMul. …
                                        -/
    a * ⅟ b = ⅟ b * (b * a * ⅟ b) := by simp [mul_assoc]
                                        /-
                                          🎉 no goals
                                        -/
                                  /-
                                    α : Type u
                                    inst✝¹ : Monoid α
                                    a b : α
                                    inst✝ : Invertible b
                                    h : Commute a b
                                    ⊢ Eq (HMul.hMul (Invertible.invOf b) (HMul.hMul (HMul.hMul b a) (Invertible.in …
                                  -/
    _ = ⅟ b * (a * b * ⅟ b) := by rw [h.eq]
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        α : Type u
                        inst✝¹ : Monoid α
                        a b : α
                        inst✝ : Invertible b
                        h : Commute a b
                        ⊢ Eq (HMul.hMul (Invertible.invOf b) (HMul.hMul (HMul.hMul a b) (Invertible.in …
                      -/
    _ = ⅟ b * a := by simp [mul_assoc]
                      /-
                        🎉 no goals
                      -/


theorem Commute.invOf_left [Monoid α] {a b : α} [Invertible b] (h : Commute b a) :
    Commute (⅟ b) a :=
  calc
                                        /-
                                          α : Type u
                                          inst✝¹ : Monoid α
                                          a b : α
                                          inst✝ : Invertible b
                                          h : Commute b a
                                          ⊢ Eq (HMul.hMul (Invertible.invOf b) a) (HMul.hMul (Invertible.invOf b) (HMul. …
                                        -/
    ⅟ b * a = ⅟ b * (a * b * ⅟ b) := by simp [mul_assoc]
                                        /-
                                          🎉 no goals
                                        -/
                                  /-
                                    α : Type u
                                    inst✝¹ : Monoid α
                                    a b : α
                                    inst✝ : Invertible b
                                    h : Commute b a
                                    ⊢ Eq (HMul.hMul (Invertible.invOf b) (HMul.hMul (HMul.hMul a b) (Invertible.in …
                                  -/
    _ = ⅟ b * (b * a * ⅟ b) := by rw [h.eq]
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        α : Type u
                        inst✝¹ : Monoid α
                        a b : α
                        inst✝ : Invertible b
                        h : Commute b a
                        ⊢ Eq (HMul.hMul (Invertible.invOf b) (HMul.hMul (HMul.hMul b a) (Invertible.in …
                      -/
    _ = a * ⅟ b := by simp [mul_assoc]
                      /-
                        🎉 no goals
                      -/


theorem commute_invOf {M : Type*} [One M] [Mul M] (m : M) [Invertible m] : Commute m (⅟ m) :=
  calc
    m * ⅟ m = 1 := mul_invOf_self m
    _ = ⅟ m * m := (invOf_mul_self m).symm


/-- This is the `Invertible` version of `Units.isUnit_units_mul` -/
abbrev invertibleOfInvertibleMul (a b : α) [Invertible a] [Invertible (a * b)] : Invertible b where
  invOf := ⅟ (a * b) * a
                       /-
                         α : Type u
                         inst✝² : Monoid α
                         a b : α
                         inst✝¹ : Invertible a
                         inst✝ : Invertible (HMul.hMul a b)
                         ⊢ Eq (HMul.hMul (HMul.hMul (Invertible.invOf (HMul.hMul a b)) a) b) 1
                       -/
  invOf_mul_self := by rw [mul_assoc, invOf_mul_self]
                       /-
                         🎉 no goals
                       -/
  mul_invOf_self := by
    rw [← (isUnit_of_invertible a).mul_right_inj, ← mul_assoc, ← mul_assoc, mul_invOf_self, mul_one,
      one_mul]


/-- This is the `Invertible` version of `Units.isUnit_mul_units` -/
abbrev invertibleOfMulInvertible (a b : α) [Invertible (a * b)] [Invertible b] : Invertible a where
  invOf := b * ⅟ (a * b)
  invOf_mul_self := by
    rw [← (isUnit_of_invertible b).mul_left_inj, mul_assoc, mul_assoc, invOf_mul_self, mul_one,
      one_mul]
                       /-
                         α : Type u
                         inst✝² : Monoid α
                         a b : α
                         inst✝¹ : Invertible (HMul.hMul a b)
                         inst✝ : Invertible b
                         ⊢ Eq (HMul.hMul a (HMul.hMul b (Invertible.invOf (HMul.hMul a b)))) 1
                       -/
  mul_invOf_self := by rw [← mul_assoc, mul_invOf_self]
                       /-
                         🎉 no goals
                       -/


/-- `invertibleOfInvertibleMul` and `invertibleMul` as an equivalence. -/
@[simps apply symm_apply]
def Invertible.mulLeft {a : α} (_ : Invertible a) (b : α) : Invertible b ≃ Invertible (a * b) where
  toFun _ := invertibleMul a b
  invFun _ := invertibleOfInvertibleMul a _
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- `invertibleOfMulInvertible` and `invertibleMul` as an equivalence. -/
@[simps apply symm_apply]
def Invertible.mulRight (a : α) {b : α} (_ : Invertible b) : Invertible a ≃ Invertible (a * b) where
  toFun _ := invertibleMul a b
  invFun _ := invertibleOfMulInvertible _ b
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


instance invertiblePow (m : α) [Invertible m] (n : ℕ) : Invertible (m ^ n) where
  invOf := ⅟ m ^ n
                       /-
                         α : Type u
                         inst✝¹ : Monoid α
                         m : α
                         inst✝ : Invertible m
                         n : Nat
                         ⊢ Eq (HMul.hMul (HPow.hPow (Invertible.invOf m) n) (HPow.hPow m n)) 1
                       -/
  invOf_mul_self := by rw [← (commute_invOf m).symm.mul_pow, invOf_mul_self, one_pow]
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u
                         inst✝¹ : Monoid α
                         m : α
                         inst✝ : Invertible m
                         n : Nat
                         ⊢ Eq (HMul.hMul (HPow.hPow m n) (HPow.hPow (Invertible.invOf m) n)) 1
                       -/
  mul_invOf_self := by rw [← (commute_invOf m).mul_pow, mul_invOf_self, one_pow]
                       /-
                         🎉 no goals
                       -/


lemma invOf_pow (m : α) [Invertible m] (n : ℕ) [Invertible (m ^ n)] : ⅟ (m ^ n) = ⅟ m ^ n :=
  @invertible_unique _ _ _ _ _ (invertiblePow m n) rfl


/-- If `x ^ n = 1` then `x` has an inverse, `x^(n - 1)`. -/
def invertibleOfPowEqOne (x : α) (n : ℕ) (hx : x ^ n = 1) (hn : n ≠ 0) : Invertible x :=
  (Units.ofPowEqOne x n hx hn).invertible


/-- Monoid homs preserve invertibility. -/
def Invertible.map {R : Type*} {S : Type*} {F : Type*} [MulOneClass R] [MulOneClass S]
    [FunLike F R S] [MonoidHomClass F R S] (f : F) (r : R) [Invertible r] :
    Invertible (f r) where
  invOf := f (⅟ r)
                       /-
                         α : Type u
                         R : Type u_1
                         S : Type u_2
                         F : Type u_3
                         inst✝⁴ : MulOneClass R
                         inst✝³ : MulOneClass S
                         inst✝² : FunLike F R S
                         inst✝¹ : MonoidHomClass F R S
                         f : F
                         r : R
                         inst✝ : Invertible r
                         ⊢ Eq (HMul.hMul (f (Invertible.invOf r)) (f r)) 1
                       -/
  invOf_mul_self := by rw [← map_mul, invOf_mul_self, map_one]
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u
                         R : Type u_1
                         S : Type u_2
                         F : Type u_3
                         inst✝⁴ : MulOneClass R
                         inst✝³ : MulOneClass S
                         inst✝² : FunLike F R S
                         inst✝¹ : MonoidHomClass F R S
                         f : F
                         r : R
                         inst✝ : Invertible r
                         ⊢ Eq (HMul.hMul (f r) (f (Invertible.invOf r))) 1
                       -/
  mul_invOf_self := by rw [← map_mul, mul_invOf_self, map_one]
                       /-
                         🎉 no goals
                       -/


/-- Note that the `Invertible (f r)` argument can be satisfied by using `letI := Invertible.map f r`
before applying this lemma. -/
theorem map_invOf {R : Type*} {S : Type*} {F : Type*} [MulOneClass R] [Monoid S]
    [FunLike F R S] [MonoidHomClass F R S] (f : F) (r : R)
    [Invertible r] [ifr : Invertible (f r)] :
    f (⅟ r) = ⅟ (f r) :=
  have h : ifr = Invertible.map f r := Subsingleton.elim _ _
     /-
       R : Type u_1
       S : Type u_2
       F : Type u_3
       inst✝⁴ : MulOneClass R
       inst✝³ : Monoid S
       inst✝² : FunLike F R S
       inst✝¹ : MonoidHomClass F R S
       f : F
       r : R
       inst✝ : Invertible r
       ifr : Invertible (f r)
       h : Eq ifr (Invertible.map f r)
       ⊢ Eq (f (Invertible.invOf r)) (Invertible.invOf (f r))
     -/
  by subst h; rfl
              /-
                🎉 no goals
              -/


/-- If a function `f : R → S` has a left-inverse that is a monoid hom,
  then `r : R` is invertible if `f r` is.

The inverse is computed as `g (⅟(f r))` -/
@[simps! (config := .lemmasOnly)]
def Invertible.ofLeftInverse {R : Type*} {S : Type*} {G : Type*} [MulOneClass R] [MulOneClass S]
    [FunLike G S R] [MonoidHomClass G S R] (f : R → S) (g : G) (r : R)
    (h : Function.LeftInverse g f) [Invertible (f r)] : Invertible r :=
  (Invertible.map g (f r)).copy _ (h r).symm


/-- Invertibility on either side of a monoid hom with a left-inverse is equivalent. -/
@[simps]
def invertibleEquivOfLeftInverse {R : Type*} {S : Type*} {F G : Type*} [Monoid R] [Monoid S]
    [FunLike F R S] [MonoidHomClass F R S] [FunLike G S R] [MonoidHomClass G S R]
    (f : F) (g : G) (r : R) (h : Function.LeftInverse g f) : Invertible (f r) ≃ Invertible r where
  toFun _ := Invertible.ofLeftInverse f _ _ h
  invFun _ := Invertible.map f _
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _

