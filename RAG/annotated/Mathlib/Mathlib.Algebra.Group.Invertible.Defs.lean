/-- `Invertible a` gives a two-sided multiplicative inverse of `a`. -/
class Invertible [Mul α] [One α] (a : α) : Type u where
  /-- The inverse of an `Invertible` element -/
  invOf : α
  /-- `invOf a` is a left inverse of `a` -/
  invOf_mul_self : invOf * a = 1
  /-- `invOf a` is a right inverse of `a` -/
  mul_invOf_self : a * invOf = 1


/-- The inverse of an `Invertible` element -/
-- This notation has the same precedence as `Inv.inv`.
prefix:max "⅟" => Invertible.invOf


@[simp]
theorem invOf_mul_self' [Mul α] [One α] (a : α) {_ : Invertible a} : ⅟ a * a = 1 :=
  Invertible.invOf_mul_self


theorem invOf_mul_self [Mul α] [One α] (a : α) [Invertible a] : ⅟ a * a = 1 := invOf_mul_self' _


@[simp]
theorem mul_invOf_self' [Mul α] [One α] (a : α) {_ : Invertible a} : a * ⅟ a = 1 :=
  Invertible.mul_invOf_self


theorem mul_invOf_self [Mul α] [One α] (a : α) [Invertible a] : a * ⅟ a = 1 := mul_invOf_self' _


@[simp]
theorem invOf_mul_cancel_left' [Monoid α] (a b : α) {_ : Invertible a} : ⅟ a * (a * b) = b := by
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    x✝ : Invertible a
    ⊢ Eq (HMul.hMul (Invertible.invOf a) (HMul.hMul a b)) b
  -/
  rw [← mul_assoc, invOf_mul_self, one_mul]
  /-
    🎉 no goals
  -/

theorem invOf_mul_cancel_left [Monoid α] (a b : α) [Invertible a] : ⅟ a * (a * b) = b :=
  invOf_mul_cancel_left' _ _


@[deprecated (since := "2024-09-07")] alias invOf_mul_self_assoc' := invOf_mul_cancel_left'

@[deprecated (since := "2024-09-07")] alias invOf_mul_self_assoc := invOf_mul_cancel_left


@[simp]
theorem mul_invOf_cancel_left' [Monoid α] (a b : α) {_ : Invertible a} : a * (⅟ a * b) = b := by
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    x✝ : Invertible a
    ⊢ Eq (HMul.hMul a (HMul.hMul (Invertible.invOf a) b)) b
  -/
  rw [← mul_assoc, mul_invOf_self, one_mul]
  /-
    🎉 no goals
  -/

theorem mul_invOf_cancel_left [Monoid α] (a b : α) [Invertible a] : a * (⅟ a * b) = b :=
  mul_invOf_cancel_left' a b


@[deprecated (since := "2024-09-07")] alias mul_invOf_self_assoc' := mul_invOf_cancel_left'

@[deprecated (since := "2024-09-07")] alias mul_invOf_self_assoc := mul_invOf_cancel_left


@[simp]
theorem invOf_mul_cancel_right' [Monoid α] (a b : α) {_ : Invertible b} : a * ⅟ b * b = a := by
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    x✝ : Invertible b
    ⊢ Eq (HMul.hMul (HMul.hMul a (Invertible.invOf b)) b) a
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/

theorem invOf_mul_cancel_right [Monoid α] (a b : α) [Invertible b] : a * ⅟ b * b = a :=
  invOf_mul_cancel_right' _ _


@[deprecated (since := "2024-09-07")] alias mul_invOf_mul_self_cancel' := invOf_mul_cancel_right'

@[deprecated (since := "2024-09-07")] alias mul_invOf_mul_self_cancel := invOf_mul_cancel_right


@[simp]
theorem mul_invOf_cancel_right' [Monoid α] (a b : α) {_ : Invertible b} : a * b * ⅟ b = a := by
  /-
    α : Type u
    inst✝ : Monoid α
    a b : α
    x✝ : Invertible b
    ⊢ Eq (HMul.hMul (HMul.hMul a b) (Invertible.invOf b)) a
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/

theorem mul_invOf_cancel_right [Monoid α] (a b : α) [Invertible b] : a * b * ⅟ b = a :=
  mul_invOf_cancel_right' _ _


@[deprecated (since := "2024-09-07")] alias mul_mul_invOf_self_cancel' := mul_invOf_cancel_right'

@[deprecated (since := "2024-09-07")] alias mul_mul_invOf_self_cancel := mul_invOf_cancel_right


theorem invOf_eq_right_inv [Monoid α] {a b : α} [Invertible a] (hac : a * b = 1) : ⅟ a = b :=
  left_inv_eq_right_inv (invOf_mul_self _) hac


theorem invOf_eq_left_inv [Monoid α] {a b : α} [Invertible a] (hac : b * a = 1) : ⅟ a = b :=
  (left_inv_eq_right_inv hac (mul_invOf_self _)).symm


theorem invertible_unique {α : Type u} [Monoid α] (a b : α) [Invertible a] [Invertible b]
    (h : a = b) : ⅟ a = ⅟ b := by
  /-
    α : Type u
    inst✝² : Monoid α
    a b : α
    inst✝¹ : Invertible a
    inst✝ : Invertible b
    h : Eq a b
    ⊢ Eq (Invertible.invOf a) (Invertible.invOf b)
  -/
  apply invOf_eq_right_inv
  /-
    case hac
    α : Type u
    inst✝² : Monoid α
    a b : α
    inst✝¹ : Invertible a
    inst✝ : Invertible b
    h : Eq a b
    ⊢ Eq (HMul.hMul a (Invertible.invOf b)) 1
  -/
  rw [h, mul_invOf_self]
  /-
    🎉 no goals
  -/


instance Invertible.subsingleton [Monoid α] (a : α) : Subsingleton (Invertible a) :=
  ⟨fun ⟨b, hba, hab⟩ ⟨c, _, hac⟩ => by
    /-
      α : Type u
      inst✝ : Monoid α
      a : α
      x✝¹ x✝ : Invertible a
      b : α
      hba : Eq (HMul.hMul b a) 1
      hab : Eq (HMul.hMul a b) 1
      c : α
      invOf_mul_self✝ : Eq (HMul.hMul c a) 1
      hac : Eq (HMul.hMul a c) 1
      ⊢ Eq { invOf := b, invOf_mul_self := hba, mul_invOf_self := hab } { invOf := c …
    -/
    congr
    /-
      case e_invOf
      α : Type u
      inst✝ : Monoid α
      a : α
      x✝¹ x✝ : Invertible a
      b : α
      hba : Eq (HMul.hMul b a) 1
      hab : Eq (HMul.hMul a b) 1
      c : α
      invOf_mul_self✝ : Eq (HMul.hMul c a) 1
      hac : Eq (HMul.hMul a c) 1
      ⊢ Eq b c
    -/
    exact left_inv_eq_right_inv hba hac⟩
    /-
      🎉 no goals
    -/


/-- If `a` is invertible and `a = b`, then `⅟a = ⅟b`. -/
@[congr]
theorem Invertible.congr [Monoid α] (a b : α) [Invertible a] [Invertible b] (h : a = b) :
                  /-
                    α : Type u
                    inst✝² : Monoid α
                    a b : α
                    inst✝¹ : Invertible a
                    inst✝ : Invertible b
                    h : Eq a b
                    ⊢ Eq (Invertible.invOf a) (Invertible.invOf b)
                  -/
    ⅟a = ⅟b := by subst h; congr; apply Subsingleton.allEq
                                  /-
                                    🎉 no goals
                                  -/


/-- If `r` is invertible and `s = r` and `si = ⅟r`, then `s` is invertible with `⅟s = si`. -/
def Invertible.copy' [MulOneClass α] {r : α} (hr : Invertible r) (s : α) (si : α) (hs : s = r)
    (hsi : si = ⅟ r) : Invertible s where
  invOf := si
                       /-
                         α : Type u
                         inst✝ : MulOneClass α
                         r : α
                         hr : Invertible r
                         s si : α
                         hs : Eq s r
                         hsi : Eq si (Invertible.invOf r)
                         ⊢ Eq (HMul.hMul si s) 1
                       -/
  invOf_mul_self := by rw [hs, hsi, invOf_mul_self]
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u
                         inst✝ : MulOneClass α
                         r : α
                         hr : Invertible r
                         s si : α
                         hs : Eq s r
                         hsi : Eq si (Invertible.invOf r)
                         ⊢ Eq (HMul.hMul s si) 1
                       -/
  mul_invOf_self := by rw [hs, hsi, mul_invOf_self]
                       /-
                         🎉 no goals
                       -/


/-- If `r` is invertible and `s = r`, then `s` is invertible. -/
abbrev Invertible.copy [MulOneClass α] {r : α} (hr : Invertible r) (s : α) (hs : s = r) :
    Invertible s :=
  hr.copy' _ _ hs rfl


/-- Each element of a group is invertible. -/
def invertibleOfGroup [Group α] (a : α) : Invertible a :=
  ⟨a⁻¹, inv_mul_cancel a, mul_inv_cancel a⟩


@[simp]
theorem invOf_eq_group_inv [Group α] (a : α) [Invertible a] : ⅟ a = a⁻¹ :=
  invOf_eq_right_inv (mul_inv_cancel a)


/-- `1` is the inverse of itself -/
def invertibleOne [Monoid α] : Invertible (1 : α) :=
  ⟨1, mul_one _, one_mul _⟩


@[simp]
theorem invOf_one' [Monoid α] {_ : Invertible (1 : α)} : ⅟ (1 : α) = 1 :=
  invOf_eq_right_inv (mul_one _)


theorem invOf_one [Monoid α] [Invertible (1 : α)] : ⅟ (1 : α) = 1 := invOf_one'


/-- `a` is the inverse of `⅟a`. -/
instance invertibleInvOf [One α] [Mul α] {a : α} [Invertible a] : Invertible (⅟ a) :=
  ⟨a, mul_invOf_self a, invOf_mul_self a⟩


@[simp]
theorem invOf_invOf [Monoid α] (a : α) [Invertible a] [Invertible (⅟ a)] : ⅟ (⅟ a) = a :=
  invOf_eq_right_inv (invOf_mul_self _)


@[simp]
theorem invOf_inj [Monoid α] {a b : α} [Invertible a] [Invertible b] : ⅟ a = ⅟ b ↔ a = b :=
  ⟨invertible_unique _ _, invertible_unique _ _⟩


/-- `⅟b * ⅟a` is the inverse of `a * b` -/
def invertibleMul [Monoid α] (a b : α) [Invertible a] [Invertible b] : Invertible (a * b) :=
                 /-
                   α : Type u
                   inst✝² : Monoid α
                   a b : α
                   inst✝¹ : Invertible a
                   inst✝ : Invertible b
                   ⊢ Eq (HMul.hMul (HMul.hMul (Invertible.invOf b) (Invertible.invOf a)) (HMul.hM …
                 -/
                 /-
                   🎉 no goals
                 -/
  ⟨⅟ b * ⅟ a, by simp [← mul_assoc], by simp [← mul_assoc]⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem invOf_mul [Monoid α] (a b : α) [Invertible a] [Invertible b] [Invertible (a * b)] :
    ⅟ (a * b) = ⅟ b * ⅟ a :=
                         /-
                           α : Type u
                           inst✝³ : Monoid α
                           a b : α
                           inst✝² : Invertible a
                           inst✝¹ : Invertible b
                           inst✝ : Invertible (HMul.hMul a b)
                           ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul (Invertible.invOf b) (Invertible.in …
                         -/
  invOf_eq_right_inv (by simp [← mul_assoc])
                         /-
                           🎉 no goals
                         -/


/-- A copy of `invertibleMul` for dot notation. -/
abbrev Invertible.mul [Monoid α] {a b : α} (_ : Invertible a) (_ : Invertible b) :
    Invertible (a * b) :=
  invertibleMul _ _


variable (c) in
theorem mul_left_inj_of_invertible : a * c = b * c ↔ a = b :=
               /-
                 α : Type u
                 inst✝¹ : Monoid α
                 a b c : α
                 inst✝ : Invertible c
                 h : Eq (HMul.hMul a c) (HMul.hMul b c)
                 ⊢ Eq a b
               -/
  ⟨fun h => by simpa using congr_arg (· * ⅟c) h, congr_arg (· * _)⟩
               /-
                 🎉 no goals
               -/


variable (c) in
theorem mul_right_inj_of_invertible : c * a = c * b ↔ a = b :=
               /-
                 α : Type u
                 inst✝¹ : Monoid α
                 a b c : α
                 inst✝ : Invertible c
                 h : Eq (HMul.hMul c a) (HMul.hMul c b)
                 ⊢ Eq a b
               -/
  ⟨fun h => by simpa using congr_arg (⅟c * ·) h, congr_arg (_ * ·)⟩
               /-
                 🎉 no goals
               -/


theorem invOf_mul_eq_iff_eq_mul_left : ⅟c * a = b ↔ a = c * b := by
  /-
    α : Type u
    inst✝¹ : Monoid α
    a b c : α
    inst✝ : Invertible c
    ⊢ Iff (Eq (HMul.hMul (Invertible.invOf c) a) b) (Eq a (HMul.hMul c b))
  -/
  rw [← mul_right_inj_of_invertible (c := c), mul_invOf_cancel_left]
  /-
    🎉 no goals
  -/


theorem mul_left_eq_iff_eq_invOf_mul : c * a = b ↔ a = ⅟c * b := by
  /-
    α : Type u
    inst✝¹ : Monoid α
    a b c : α
    inst✝ : Invertible c
    ⊢ Iff (Eq (HMul.hMul c a) b) (Eq a (HMul.hMul (Invertible.invOf c) b))
  -/
  rw [← mul_right_inj_of_invertible (c := ⅟c), invOf_mul_cancel_left]
  /-
    🎉 no goals
  -/


theorem mul_invOf_eq_iff_eq_mul_right : a * ⅟c = b ↔ a = b * c := by
  /-
    α : Type u
    inst✝¹ : Monoid α
    a b c : α
    inst✝ : Invertible c
    ⊢ Iff (Eq (HMul.hMul a (Invertible.invOf c)) b) (Eq a (HMul.hMul b c))
  -/
  rw [← mul_left_inj_of_invertible (c := c), invOf_mul_cancel_right]
  /-
    🎉 no goals
  -/


theorem mul_right_eq_iff_eq_mul_invOf : a * c = b ↔ a = b * ⅟c := by
  /-
    α : Type u
    inst✝¹ : Monoid α
    a b c : α
    inst✝ : Invertible c
    ⊢ Iff (Eq (HMul.hMul a c) b) (Eq a (HMul.hMul b (Invertible.invOf c)))
  -/
  rw [← mul_left_inj_of_invertible (c := ⅟c), mul_invOf_cancel_right]
  /-
    🎉 no goals
  -/


