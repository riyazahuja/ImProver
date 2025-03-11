@[to_additive (attr := ext)]
theorem Monoid.ext {M : Type u} ⦃m₁ m₂ : Monoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M)) :
    m₁ = m₂ := by
  /-
    M : Type u
    m₁ m₂ : Monoid M
    h_mul : Eq HMul.hMul HMul.hMul
    ⊢ Eq m₁ m₂
  -/
  have : m₁.toMulOneClass = m₂.toMulOneClass := MulOneClass.ext h_mul
  /-
    M : Type u
    m₁ m₂ : Monoid M
    h_mul : Eq HMul.hMul HMul.hMul
    this : Eq Monoid.toMulOneClass Monoid.toMulOneClass
    ⊢ Eq m₁ m₂
  -/
  have h₁ : m₁.one = m₂.one := congr_arg (·.one) this
  let f : @MonoidHom M M m₁.toMulOneClass m₂.toMulOneClass :=
    @MonoidHom.mk _ _ (_) _ (@OneHom.mk _ _ (_) _ id h₁)
      (fun x y => congr_fun (congr_fun h_mul x) y)
  have : m₁.npow = m₂.npow := by
    ext n x
    exact @MonoidHom.map_pow M M m₁ m₂ f x n
  /-
    M : Type u
    m₁ m₂ : Monoid M
    h_mul : Eq HMul.hMul HMul.hMul
    this✝ : Eq Monoid.toMulOneClass Monoid.toMulOneClass
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this : Eq Monoid.npow Monoid.npow
    ⊢ Eq m₁ m₂
  -/
  rcases m₁ with @⟨@⟨⟨_⟩⟩, ⟨_⟩⟩
  /-
    case mk.mk.mk.mk
    M : Type u
    m₂ : Monoid M
    npow✝ : Nat → M → M
    mul✝ : M → M → M
    mul_assoc✝ : ∀ (a b c : M), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMu …
    npow_succ✝ : ∀ (n : Nat) (x : M), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    one✝ : M
    npow_zero✝ : ∀ (x : M), Eq (npow✝ 0 x) 1
    one_mul✝ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : M), Eq (HMul.hMul a 1) a
    h_mul : Eq HMul.hMul HMul.hMul
    this✝ : Eq Monoid.toMulOneClass Monoid.toMulOneClass
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this : Eq Monoid.npow Monoid.npow
    ⊢ Eq (Monoid.mk one_mul✝ mul_one✝ npow✝ npow_zero✝ npow_succ✝) m₂
  -/
  rcases m₂ with @⟨@⟨⟨_⟩⟩, ⟨_⟩⟩
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    M : Type u
    npow✝¹ : Nat → M → M
    mul✝¹ : M → M → M
    mul_assoc✝¹ : ∀ (a b c : M), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HM …
    npow_succ✝¹ : ∀ (n : Nat) (x : M), Eq (npow✝¹ (HAdd.hAdd n 1) x) (HMul.hMul (n …
    one✝¹ : M
    npow_zero✝¹ : ∀ (x : M), Eq (npow✝¹ 0 x) 1
    one_mul✝¹ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝ : Nat → M → M
    mul✝ : M → M → M
    mul_assoc✝ : ∀ (a b c : M), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMu …
    npow_succ✝ : ∀ (n : Nat) (x : M), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    one✝ : M
    npow_zero✝ : ∀ (x : M), Eq (npow✝ 0 x) 1
    one_mul✝ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : M), Eq (HMul.hMul a 1) a
    h_mul : Eq HMul.hMul HMul.hMul
    this✝ : Eq Monoid.toMulOneClass Monoid.toMulOneClass
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this : Eq Monoid.npow Monoid.npow
    ⊢ Eq (Monoid.mk one_mul✝¹ mul_one✝¹ npow✝¹ npow_zero✝¹ npow_succ✝¹) (Monoid.mk …
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive]
theorem CommMonoid.toMonoid_injective {M : Type u} :
    Function.Injective (@CommMonoid.toMonoid M) := by
  /-
    M : Type u
    ⊢ Function.Injective (@CommMonoid.toMonoid M)
  -/
  rintro ⟨⟩ ⟨⟩ h
  /-
    case mk.mk
    M : Type u
    toMonoid✝¹ : Monoid M
    mul_comm✝¹ : ∀ (a b : M), Eq (HMul.hMul a b) (HMul.hMul b a)
    toMonoid✝ : Monoid M
    mul_comm✝ : ∀ (a b : M), Eq (HMul.hMul a b) (HMul.hMul b a)
    h : Eq CommMonoid.toMonoid CommMonoid.toMonoid
    ⊢ Eq (CommMonoid.mk mul_comm✝¹) (CommMonoid.mk mul_comm✝)
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive (attr := ext)]
theorem CommMonoid.ext {M : Type*} ⦃m₁ m₂ : CommMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M)) : m₁ = m₂ :=
  CommMonoid.toMonoid_injective <| Monoid.ext h_mul


@[to_additive]
theorem LeftCancelMonoid.toMonoid_injective {M : Type u} :
    Function.Injective (@LeftCancelMonoid.toMonoid M) := by
  /-
    M : Type u
    ⊢ Function.Injective (@LeftCancelMonoid.toMonoid M)
  -/
  rintro @⟨@⟨⟩⟩ @⟨@⟨⟩⟩ h
  /-
    case mk.mk.mk.mk
    M : Type u
    toSemigroup✝¹ : Semigroup M
    toOne✝¹ : One M
    one_mul✝¹ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝¹ : Nat → M → M
    npow_zero✝¹ : ∀ (x : M), Eq (npow✝¹ 0 x) 1
    npow_succ✝¹ : ∀ (n : Nat) (x : M), Eq (npow✝¹ (HAdd.hAdd n 1) x) (HMul.hMul (n …
    mul_left_cancel✝¹ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
    toSemigroup✝ : Semigroup M
    toOne✝ : One M
    one_mul✝ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝ : Nat → M → M
    npow_zero✝ : ∀ (x : M), Eq (npow✝ 0 x) 1
    npow_succ✝ : ∀ (n : Nat) (x : M), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    mul_left_cancel✝ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
    h : Eq LeftCancelMonoid.toMonoid LeftCancelMonoid.toMonoid
    ⊢ Eq (LeftCancelMonoid.mk mul_left_cancel✝¹) (LeftCancelMonoid.mk mul_left_can …
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  congr <;> injection h
            /-
              🎉 no goals
            -/


@[to_additive (attr := ext)]
theorem LeftCancelMonoid.ext {M : Type u} ⦃m₁ m₂ : LeftCancelMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M)) :
    m₁ = m₂ :=
  LeftCancelMonoid.toMonoid_injective <| Monoid.ext h_mul


@[to_additive]
theorem RightCancelMonoid.toMonoid_injective {M : Type u} :
    Function.Injective (@RightCancelMonoid.toMonoid M) := by
  /-
    M : Type u
    ⊢ Function.Injective (@RightCancelMonoid.toMonoid M)
  -/
  rintro @⟨@⟨⟩⟩ @⟨@⟨⟩⟩ h
  /-
    case mk.mk.mk.mk
    M : Type u
    toSemigroup✝¹ : Semigroup M
    toOne✝¹ : One M
    one_mul✝¹ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝¹ : Nat → M → M
    npow_zero✝¹ : ∀ (x : M), Eq (npow✝¹ 0 x) 1
    npow_succ✝¹ : ∀ (n : Nat) (x : M), Eq (npow✝¹ (HAdd.hAdd n 1) x) (HMul.hMul (n …
    mul_right_cancel✝¹ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
    toSemigroup✝ : Semigroup M
    toOne✝ : One M
    one_mul✝ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝ : Nat → M → M
    npow_zero✝ : ∀ (x : M), Eq (npow✝ 0 x) 1
    npow_succ✝ : ∀ (n : Nat) (x : M), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    mul_right_cancel✝ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
    h : Eq RightCancelMonoid.toMonoid RightCancelMonoid.toMonoid
    ⊢ Eq (RightCancelMonoid.mk mul_right_cancel✝¹) (RightCancelMonoid.mk mul_right …
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  congr <;> injection h
            /-
              🎉 no goals
            -/


@[to_additive (attr := ext)]
theorem RightCancelMonoid.ext {M : Type u} ⦃m₁ m₂ : RightCancelMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M))  :
    m₁ = m₂ :=
  RightCancelMonoid.toMonoid_injective <| Monoid.ext h_mul


@[to_additive]
theorem CancelMonoid.toLeftCancelMonoid_injective {M : Type u} :
    Function.Injective (@CancelMonoid.toLeftCancelMonoid M) := by
  /-
    M : Type u
    ⊢ Function.Injective (@CancelMonoid.toLeftCancelMonoid M)
  -/
  rintro ⟨⟩ ⟨⟩ h
  /-
    case mk.mk
    M : Type u
    toLeftCancelMonoid✝¹ : LeftCancelMonoid M
    mul_right_cancel✝¹ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
    toLeftCancelMonoid✝ : LeftCancelMonoid M
    mul_right_cancel✝ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul c b) → Eq a c
    h : Eq CancelMonoid.toLeftCancelMonoid CancelMonoid.toLeftCancelMonoid
    ⊢ Eq (CancelMonoid.mk mul_right_cancel✝¹) (CancelMonoid.mk mul_right_cancel✝)
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive (attr := ext)]
theorem CancelMonoid.ext {M : Type*} ⦃m₁ m₂ : CancelMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M)) :
    m₁ = m₂ :=
  CancelMonoid.toLeftCancelMonoid_injective <| LeftCancelMonoid.ext h_mul


@[to_additive]
theorem CancelCommMonoid.toCommMonoid_injective {M : Type u} :
    Function.Injective (@CancelCommMonoid.toCommMonoid M) := by
  /-
    M : Type u
    ⊢ Function.Injective (@CancelCommMonoid.toCommMonoid M)
  -/
  rintro @⟨@⟨@⟨⟩⟩⟩ @⟨@⟨@⟨⟩⟩⟩ h
  /-
    case mk.mk.mk.mk.mk.mk
    M : Type u
    toSemigroup✝¹ : Semigroup M
    toOne✝¹ : One M
    one_mul✝¹ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝¹ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝¹ : Nat → M → M
    npow_zero✝¹ : ∀ (x : M), Eq (npow✝¹ 0 x) 1
    npow_succ✝¹ : ∀ (n : Nat) (x : M), Eq (npow✝¹ (HAdd.hAdd n 1) x) (HMul.hMul (n …
    mul_comm✝¹ : ∀ (a b : M), Eq (HMul.hMul a b) (HMul.hMul b a)
    mul_left_cancel✝¹ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
    toSemigroup✝ : Semigroup M
    toOne✝ : One M
    one_mul✝ : ∀ (a : M), Eq (HMul.hMul 1 a) a
    mul_one✝ : ∀ (a : M), Eq (HMul.hMul a 1) a
    npow✝ : Nat → M → M
    npow_zero✝ : ∀ (x : M), Eq (npow✝ 0 x) 1
    npow_succ✝ : ∀ (n : Nat) (x : M), Eq (npow✝ (HAdd.hAdd n 1) x) (HMul.hMul (npo …
    mul_comm✝ : ∀ (a b : M), Eq (HMul.hMul a b) (HMul.hMul b a)
    mul_left_cancel✝ : ∀ (a b c : M), Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
    h : Eq CancelCommMonoid.toCommMonoid CancelCommMonoid.toCommMonoid
    ⊢ Eq (CancelCommMonoid.mk mul_left_cancel✝¹) (CancelCommMonoid.mk mul_left_can …
  -/
  congr <;> {
    injection h with h'
    injection h' }


@[to_additive (attr := ext)]
theorem CancelCommMonoid.ext {M : Type*} ⦃m₁ m₂ : CancelCommMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M)) :
    m₁ = m₂ :=
  CancelCommMonoid.toCommMonoid_injective <| CommMonoid.ext h_mul


@[to_additive (attr := ext)]
theorem DivInvMonoid.ext {M : Type*} ⦃m₁ m₂ : DivInvMonoid M⦄
    (h_mul : (letI := m₁; HMul.hMul : M → M → M) = (letI := m₂; HMul.hMul : M → M → M))
    (h_inv : (letI := m₁; Inv.inv : M → M) = (letI := m₂; Inv.inv : M → M)) : m₁ = m₂ := by
  /-
    M : Type u_1
    m₁ m₂ : DivInvMonoid M
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    ⊢ Eq m₁ m₂
  -/
  have h_mon := Monoid.ext h_mul
  /-
    M : Type u_1
    m₁ m₂ : DivInvMonoid M
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    h_mon : Eq DivInvMonoid.toMonoid DivInvMonoid.toMonoid
    ⊢ Eq m₁ m₂
  -/
  have h₁ : m₁.one = m₂.one := congr_arg (·.one) h_mon
  let f : @MonoidHom M M m₁.toMulOneClass m₂.toMulOneClass :=
    @MonoidHom.mk _ _ (_) _ (@OneHom.mk _ _ (_) _ id h₁)
      (fun x y => congr_fun (congr_fun h_mul x) y)
  /-
    M : Type u_1
    m₁ m₂ : DivInvMonoid M
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    h_mon : Eq DivInvMonoid.toMonoid DivInvMonoid.toMonoid
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    ⊢ Eq m₁ m₂
  -/
  have : m₁.npow = m₂.npow := congr_arg (·.npow) h_mon
  have : m₁.zpow = m₂.zpow := by
    ext m x
    exact @MonoidHom.map_zpow' M M m₁ m₂ f (congr_fun h_inv) x m
  have : m₁.div = m₂.div := by
    ext a b
    exact @map_div' _ _
      (F := @MonoidHom _ _ (_) _) _ (id _) _
      (@MonoidHom.instMonoidHomClass _ _ (_) _) f (congr_fun h_inv) a b
  /-
    M : Type u_1
    m₁ m₂ : DivInvMonoid M
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    h_mon : Eq DivInvMonoid.toMonoid DivInvMonoid.toMonoid
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this✝¹ : Eq Monoid.npow Monoid.npow
    this✝ : Eq DivInvMonoid.zpow DivInvMonoid.zpow
    this : Eq Div.div Div.div
    ⊢ Eq m₁ m₂
  -/
  rcases m₁ with @⟨_, ⟨_⟩, ⟨_⟩⟩
  /-
    case mk.mk.mk
    M : Type u_1
    m₂ : DivInvMonoid M
    toMonoid✝ : Monoid M
    zpow✝ : Int → M → M
    zpow_zero'✝ : ∀ (a : M), Eq (zpow✝ 0 a) 1
    zpow_succ'✝ : ∀ (n : Nat) (a : M), Eq (zpow✝ (↑n.succ) a) (HMul.hMul (zpow✝ (↑ …
    inv✝ : M → M
    zpow_neg'✝ : ∀ (n : Nat) (a : M), Eq (zpow✝ (Int.negSucc n) a) (Inv.inv (zpow✝ …
    div✝ : M → M → M
    div_eq_mul_inv✝ : ∀ (a b : M), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    h_mon : Eq DivInvMonoid.toMonoid DivInvMonoid.toMonoid
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this✝¹ : Eq Monoid.npow Monoid.npow
    this✝ : Eq DivInvMonoid.zpow DivInvMonoid.zpow
    this : Eq Div.div Div.div
    ⊢ Eq (DivInvMonoid.mk div_eq_mul_inv✝ zpow✝ zpow_zero'✝ zpow_succ'✝ zpow_neg'✝ …
  -/
  rcases m₂ with @⟨_, ⟨_⟩, ⟨_⟩⟩
  /-
    case mk.mk.mk.mk.mk.mk
    M : Type u_1
    toMonoid✝¹ : Monoid M
    zpow✝¹ : Int → M → M
    zpow_zero'✝¹ : ∀ (a : M), Eq (zpow✝¹ 0 a) 1
    zpow_succ'✝¹ : ∀ (n : Nat) (a : M), Eq (zpow✝¹ (↑n.succ) a) (HMul.hMul (zpow✝¹ …
    inv✝¹ : M → M
    zpow_neg'✝¹ : ∀ (n : Nat) (a : M), Eq (zpow✝¹ (Int.negSucc n) a) (Inv.inv (zpo …
    div✝¹ : M → M → M
    div_eq_mul_inv✝¹ : ∀ (a b : M), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
    toMonoid✝ : Monoid M
    zpow✝ : Int → M → M
    zpow_zero'✝ : ∀ (a : M), Eq (zpow✝ 0 a) 1
    zpow_succ'✝ : ∀ (n : Nat) (a : M), Eq (zpow✝ (↑n.succ) a) (HMul.hMul (zpow✝ (↑ …
    inv✝ : M → M
    zpow_neg'✝ : ∀ (n : Nat) (a : M), Eq (zpow✝ (Int.negSucc n) a) (Inv.inv (zpow✝ …
    div✝ : M → M → M
    div_eq_mul_inv✝ : ∀ (a b : M), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
    h_mul : Eq HMul.hMul HMul.hMul
    h_inv : Eq Inv.inv Inv.inv
    h_mon : Eq DivInvMonoid.toMonoid DivInvMonoid.toMonoid
    h₁ : Eq One.one One.one
    f : MonoidHom M M := { toFun := id, map_one' := h₁, map_mul' := ⋯ }
    this✝¹ : Eq Monoid.npow Monoid.npow
    this✝ : Eq DivInvMonoid.zpow DivInvMonoid.zpow
    this : Eq Div.div Div.div
    ⊢ Eq (DivInvMonoid.mk div_eq_mul_inv✝¹ zpow✝¹ zpow_zero'✝¹ zpow_succ'✝¹ zpow_n …
  -/
  congr
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Group.toDivInvMonoid_injective {G : Type*} : Injective (@Group.toDivInvMonoid G) := by
  /-
    G : Type u_1
    ⊢ Function.Injective (@Group.toDivInvMonoid G)
  -/
  rintro ⟨⟩ ⟨⟩ ⟨⟩; rfl
                   /-
                     🎉 no goals
                   -/


@[to_additive (attr := ext)]
theorem Group.ext {G : Type*} ⦃g₁ g₂ : Group G⦄ (h_mul : g₁.mul = g₂.mul) : g₁ = g₂ := by
  /-
    G : Type u_1
    g₁ g₂ : Group G
    h_mul : Eq Mul.mul Mul.mul
    ⊢ Eq g₁ g₂
  -/
  have h₁ : g₁.one = g₂.one := congr_arg (·.one) (Monoid.ext h_mul)
  let f : @MonoidHom G G g₁.toMulOneClass g₂.toMulOneClass :=
    @MonoidHom.mk _ _ (_) _ (@OneHom.mk _ _ (_) _ id h₁)
      (fun x y => congr_fun (congr_fun h_mul x) y)
  exact
    Group.toDivInvMonoid_injective
      (DivInvMonoid.ext h_mul
        (funext <| @MonoidHom.map_inv G G g₁ g₂.toDivisionMonoid f))


@[to_additive]
lemma CommGroup.toGroup_injective {G : Type*} : Injective (@CommGroup.toGroup G) := by
  /-
    G : Type u_1
    ⊢ Function.Injective (@CommGroup.toGroup G)
  -/
  rintro ⟨⟩ ⟨⟩ ⟨⟩; rfl
                   /-
                     🎉 no goals
                   -/


@[to_additive (attr := ext)]
theorem CommGroup.ext {G : Type*} ⦃g₁ g₂ : CommGroup G⦄ (h_mul : g₁.mul = g₂.mul) : g₁ = g₂ :=
  CommGroup.toGroup_injective <| Group.ext h_mul

