@[to_additive]
theorem toMap_injective_iff
    {M N : Type*} [CommMonoid M] {S : Submonoid M} [CommMonoid N] (f : LocalizationMap S N) :
    Injective (LocalizationMap.toMap f) ↔ ∀ ⦃x⦄, x ∈ S → IsLeftRegular x := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : CommMonoid M
    S : Submonoid M
    inst✝ : CommMonoid N
    f : S.LocalizationMap N
    ⊢ Iff (Function.Injective ⇑f.toMap) (∀ ⦃x : M⦄, Membership.mem S x → IsLeftReg …
  -/
  rw [Injective]
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : CommMonoid M
    S : Submonoid M
    inst✝ : CommMonoid N
    f : S.LocalizationMap N
    ⊢ Iff (∀ ⦃a₁ a₂ : M⦄, Eq (f.toMap a₁) (f.toMap a₂) → Eq a₁ a₂) (∀ ⦃x : M⦄, Mem …
  -/
  constructor <;> intro h
    /-
      case mp
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃a₁ a₂ : M⦄, Eq (f.toMap a₁) (f.toMap a₂) → Eq a₁ a₂
      ⊢ ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    -/
  · intro x hx y z hyz
    /-
      case mp
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃a₁ a₂ : M⦄, Eq (f.toMap a₁) (f.toMap a₂) → Eq a₁ a₂
      x : M
      hx : Membership.mem S x
      y z : M
      hyz : Eq ((fun x_1 => HMul.hMul x x_1) y) ((fun x_1 => HMul.hMul x x_1) z)
      ⊢ Eq y z
    -/
    simp_rw [LocalizationMap.eq_iff_exists] at h
    /-
      case mp
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      x : M
      hx : Membership.mem S x
      y z : M
      hyz : Eq ((fun x_1 => HMul.hMul x x_1) y) ((fun x_1 => HMul.hMul x x_1) z)
      h : ∀ ⦃a₁ a₂ : M⦄, (Exists fun c => Eq (HMul.hMul (↑c) a₁) (HMul.hMul (↑c) a₂) …
      ⊢ Eq y z
    -/
    apply (fun y z _ => h) y z x
    /-
      case mp.a
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      x : M
      hx : Membership.mem S x
      y z : M
      hyz : Eq ((fun x_1 => HMul.hMul x x_1) y) ((fun x_1 => HMul.hMul x x_1) z)
      h : ∀ ⦃a₁ a₂ : M⦄, (Exists fun c => Eq (HMul.hMul (↑c) a₁) (HMul.hMul (↑c) a₂) …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) y) (HMul.hMul (↑c) z)
    -/
    lift x to S using hx
    /-
      case mp.a.intro
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      y z : M
      h : ∀ ⦃a₁ a₂ : M⦄, (Exists fun c => Eq (HMul.hMul (↑c) a₁) (HMul.hMul (↑c) a₂) …
      x : Subtype fun x => Membership.mem S x
      hyz : Eq ((fun x_1 => HMul.hMul (↑x) x_1) y) ((fun x_1 => HMul.hMul (↑x) x_1) z)
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) y) (HMul.hMul (↑c) z)
    -/
    use x
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
      ⊢ ∀ ⦃a₁ a₂ : M⦄, Eq (f.toMap a₁) (f.toMap a₂) → Eq a₁ a₂
    -/
  · intro a b hab
    /-
      case mpr
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
      a b : M
      hab : Eq (f.toMap a) (f.toMap b)
      ⊢ Eq a b
    -/
    rw [LocalizationMap.eq_iff_exists] at hab
    /-
      case mpr
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
      a b : M
      hab : Exists fun c => Eq (HMul.hMul (↑c) a) (HMul.hMul (↑c) b)
      ⊢ Eq a b
    -/
    obtain ⟨c,hc⟩ := hab
    /-
      case mpr.intro
      M : Type u_1
      N : Type u_2
      inst✝¹ : CommMonoid M
      S : Submonoid M
      inst✝ : CommMonoid N
      f : S.LocalizationMap N
      h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
      a b : M
      c : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul (↑c) a) (HMul.hMul (↑c) b)
      ⊢ Eq a b
    -/
    apply (fun x a => h a) c (SetLike.coe_mem c) hc
    /-
      🎉 no goals
    -/


variable {S N} in
/-- If `S` contains `0` then the localization at `S` is trivial. -/
theorem LocalizationMap.subsingleton (f : Submonoid.LocalizationMap S N) (h : 0 ∈ S) :
    Subsingleton N := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝ : CommMonoidWithZero N
    f : S.LocalizationMap N
    h : Membership.mem S 0
    ⊢ Subsingleton N
  -/
  refine ⟨fun a b ↦ ?_⟩
  /-
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝ : CommMonoidWithZero N
    f : S.LocalizationMap N
    h : Membership.mem S 0
    a b : N
    ⊢ Eq a b
  -/
  rw [← LocalizationMap.mk'_sec f a, ← LocalizationMap.mk'_sec f b, LocalizationMap.eq]
  /-
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝ : CommMonoidWithZero N
    f : S.LocalizationMap N
    h : Membership.mem S 0
    a b : N
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑(f.sec b).2) (f.sec a).1)) ( …
  -/
  exact ⟨⟨0, h⟩, by simp only [zero_mul]⟩
  /-
    🎉 no goals
  -/


/-- The type of homomorphisms between monoids with zero satisfying the characteristic predicate:
if `f : M →*₀ N` satisfies this predicate, then `N` is isomorphic to the localization of `M` at
`S`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure LocalizationWithZeroMap extends LocalizationMap S N where
  map_zero' : toFun 0 = 0

-- Porting note: no docstrings for LocalizationWithZeroMap.map_zero'

/-- The monoid with zero hom underlying a `LocalizationMap`. -/
def LocalizationWithZeroMap.toMonoidWithZeroHom (f : LocalizationWithZeroMap S N) : M →*₀ N :=
  { f with }


theorem mk_zero (x : S) : mk 0 (x : S) = 0 := OreLocalization.zero_oreDiv' _


instance : CommMonoidWithZero (Localization S) where
  zero_mul := fun x ↦ Localization.induction_on x fun y => by
    /-
      M : Type u_1
      inst✝² : CommMonoidWithZero M
      S : Submonoid M
      N : Type u_2
      inst✝¹ : CommMonoidWithZero N
      P : Type u_3
      inst✝ : CommMonoidWithZero P
      x : Localization S
      y : Prod M (Subtype fun x => Membership.mem S x)
      ⊢ Eq (HMul.hMul 0 (Localization.mk y.1 y.2)) 0
    -/
    simp only [← Localization.mk_zero y.2, mk_mul, mk_eq_mk_iff, mul_zero, zero_mul, r_of_eq]
    /-
      🎉 no goals
    -/
  mul_zero := fun x ↦ Localization.induction_on x fun y => by
    /-
      M : Type u_1
      inst✝² : CommMonoidWithZero M
      S : Submonoid M
      N : Type u_2
      inst✝¹ : CommMonoidWithZero N
      P : Type u_3
      inst✝ : CommMonoidWithZero P
      x : Localization S
      y : Prod M (Subtype fun x => Membership.mem S x)
      ⊢ Eq (HMul.hMul (Localization.mk y.1 y.2) 0) 0
    -/
    simp only [← Localization.mk_zero y.2, mk_mul, mk_eq_mk_iff, mul_zero, zero_mul, r_of_eq]
    /-
      🎉 no goals
    -/


theorem liftOn_zero {p : Type*} (f : M → S → p) (H) : liftOn 0 f H = f 0 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    S : Submonoid M
    p : Type u_4
    f : M → (Subtype fun x => Membership.mem S x) → p
    H : ∀ {a c : M} {b d : Subtype fun x => Membership.mem S x}, (Localization.r S …
    ⊢ Eq (Localization.liftOn 0 f H) (f 0 1)
  -/
  rw [← mk_zero 1, liftOn_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem LocalizationMap.sec_zero_fst {f : LocalizationMap S N} : f.toMap (f.sec 0).fst = 0 := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝ : CommMonoidWithZero N
    f : S.LocalizationMap N
    ⊢ Eq (f.toMap (f.sec 0).1) 0
  -/
  rw [LocalizationMap.sec_spec', mul_zero]
  /-
    🎉 no goals
  -/


/-- Given a Localization map `f : M →*₀ N` for a Submonoid `S ⊆ M` and a map of
`CommMonoidWithZero`s `g : M →*₀ P` such that `g y` is invertible for all `y : S`, the
homomorphism induced from `N` to `P` sending `z : N` to `g x * (g y)⁻¹`, where `(x, y) : M × S`
are such that `z = f x * (f y)⁻¹`. -/
noncomputable def lift (f : LocalizationWithZeroMap S N) (g : M →*₀ P)
    (hg : ∀ y : S, IsUnit (g y)) : N →*₀ P :=
  { @LocalizationMap.lift _ _ _ _ _ _ _ f.toLocalizationMap g.toMonoidHom hg with
    map_zero' := by
      /-
        M : Type u_1
        inst✝² : CommMonoidWithZero M
        S : Submonoid M
        N : Type u_2
        inst✝¹ : CommMonoidWithZero N
        P : Type u_3
        inst✝ : CommMonoidWithZero P
        f : S.LocalizationWithZeroMap N
        g : MonoidWithZeroHom M P
        hg : ∀ (y : Subtype fun x => Membership.mem S x), IsUnit (g ↑y)
        ⊢ Eq ((↑__src✝).toFun 0) 0
      -/
      erw [LocalizationMap.lift_spec f.toLocalizationMap hg 0 0]
      /-
        M : Type u_1
        inst✝² : CommMonoidWithZero M
        S : Submonoid M
        N : Type u_2
        inst✝¹ : CommMonoidWithZero N
        P : Type u_3
        inst✝ : CommMonoidWithZero P
        f : S.LocalizationWithZeroMap N
        g : MonoidWithZeroHom M P
        hg : ∀ (y : Subtype fun x => Membership.mem S x), IsUnit (g ↑y)
        ⊢ Eq (↑g (f.sec 0).1) (HMul.hMul (↑g ↑(f.sec 0).2) 0)
      -/
      rw [mul_zero, ← map_zero g, ← g.toMonoidHom_coe]
      /-
        M : Type u_1
        inst✝² : CommMonoidWithZero M
        S : Submonoid M
        N : Type u_2
        inst✝¹ : CommMonoidWithZero N
        P : Type u_3
        inst✝ : CommMonoidWithZero P
        f : S.LocalizationWithZeroMap N
        g : MonoidWithZeroHom M P
        hg : ∀ (y : Subtype fun x => Membership.mem S x), IsUnit (g ↑y)
        ⊢ Eq (↑g (f.sec 0).1) ((↑↑g).toFun 0)
      -/
      refine f.toLocalizationMap.eq_of_eq hg ?_
      /-
        M : Type u_1
        inst✝² : CommMonoidWithZero M
        S : Submonoid M
        N : Type u_2
        inst✝¹ : CommMonoidWithZero N
        P : Type u_3
        inst✝ : CommMonoidWithZero P
        f : S.LocalizationWithZeroMap N
        g : MonoidWithZeroHom M P
        hg : ∀ (y : Subtype fun x => Membership.mem S x), IsUnit (g ↑y)
        ⊢ Eq (f.toMap (f.sec 0).1) (f.toMap 0)
      -/
      rw [LocalizationMap.sec_zero_fst]
      /-
        M : Type u_1
        inst✝² : CommMonoidWithZero M
        S : Submonoid M
        N : Type u_2
        inst✝¹ : CommMonoidWithZero N
        P : Type u_3
        inst✝ : CommMonoidWithZero P
        f : S.LocalizationWithZeroMap N
        g : MonoidWithZeroHom M P
        hg : ∀ (y : Subtype fun x => Membership.mem S x), IsUnit (g ↑y)
        ⊢ Eq 0 (f.toMap 0)
      -/
      exact f.toMonoidWithZeroHom.map_zero.symm }
      /-
        🎉 no goals
      -/


/-- Given a Localization map `f : M →*₀ N` for a Submonoid `S ⊆ M`,
if `M` is left cancellative monoid with zero, and all elements of `S` are
left regular, then N is a left cancellative monoid with zero. -/
theorem leftCancelMulZero_of_le_isLeftRegular
    (f : LocalizationWithZeroMap S N) [IsLeftCancelMulZero M]
    (h : ∀ ⦃x⦄, x ∈ S → IsLeftRegular x) : IsLeftCancelMulZero N := by
  /-
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    ⊢ IsLeftCancelMulZero N
  -/
  let fl := f.toLocalizationMap
  /-
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    ⊢ IsLeftCancelMulZero N
  -/
  let g := f.toMap
  /-
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    ⊢ IsLeftCancelMulZero N
  -/
  constructor
  /-
    case mul_left_cancel_of_ne_zero
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    ⊢ ∀ {a b c : N}, Ne a 0 → Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
  -/
  intro a z w ha hazw
  /-
    case mul_left_cancel_of_ne_zero
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    a z w : N
    ha : Ne a 0
    hazw : Eq (HMul.hMul a z) (HMul.hMul a w)
    ⊢ Eq z w
  -/
  obtain ⟨b, hb⟩ := LocalizationMap.surj fl a
  /-
    case mul_left_cancel_of_ne_zero.intro
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    a z w : N
    ha : Ne a 0
    hazw : Eq (HMul.hMul a z) (HMul.hMul a w)
    b : Prod M (Subtype fun x => Membership.mem S x)
    hb : Eq (HMul.hMul a (fl.toMap ↑b.2)) (fl.toMap b.1)
    ⊢ Eq z w
  -/
  obtain ⟨x, hx⟩ := LocalizationMap.surj fl z
  /-
    case mul_left_cancel_of_ne_zero.intro.intro
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    a z w : N
    ha : Ne a 0
    hazw : Eq (HMul.hMul a z) (HMul.hMul a w)
    b : Prod M (Subtype fun x => Membership.mem S x)
    hb : Eq (HMul.hMul a (fl.toMap ↑b.2)) (fl.toMap b.1)
    x : Prod M (Subtype fun x => Membership.mem S x)
    hx : Eq (HMul.hMul z (fl.toMap ↑x.2)) (fl.toMap x.1)
    ⊢ Eq z w
  -/
  obtain ⟨y, hy⟩ := LocalizationMap.surj fl w
  rw [(LocalizationMap.eq_mk'_iff_mul_eq fl).mpr hx,
    (LocalizationMap.eq_mk'_iff_mul_eq fl).mpr hy, LocalizationMap.eq]
  /-
    case mul_left_cancel_of_ne_zero.intro.intro.intro
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    a z w : N
    ha : Ne a 0
    hazw : Eq (HMul.hMul a z) (HMul.hMul a w)
    b : Prod M (Subtype fun x => Membership.mem S x)
    hb : Eq (HMul.hMul a (fl.toMap ↑b.2)) (fl.toMap b.1)
    x : Prod M (Subtype fun x => Membership.mem S x)
    hx : Eq (HMul.hMul z (fl.toMap ↑x.2)) (fl.toMap x.1)
    y : Prod M (Subtype fun x => Membership.mem S x)
    hy : Eq (HMul.hMul w (fl.toMap ↑y.2)) (fl.toMap y.1)
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑y.2) x.1)) (HMul.hMul (↑c) ( …
  -/
  use 1
  /-
    case h
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsLeftCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsLeftRegular x
    fl : S.LocalizationMap N := f.toLocalizationMap
    g : MonoidHom M N := f.toMap
    a z w : N
    ha : Ne a 0
    hazw : Eq (HMul.hMul a z) (HMul.hMul a w)
    b : Prod M (Subtype fun x => Membership.mem S x)
    hb : Eq (HMul.hMul a (fl.toMap ↑b.2)) (fl.toMap b.1)
    x : Prod M (Subtype fun x => Membership.mem S x)
    hx : Eq (HMul.hMul z (fl.toMap ↑x.2)) (fl.toMap x.1)
    y : Prod M (Subtype fun x => Membership.mem S x)
    hy : Eq (HMul.hMul w (fl.toMap ↑y.2)) (fl.toMap y.1)
    ⊢ Eq (HMul.hMul (↑1) (HMul.hMul (↑y.2) x.1)) (HMul.hMul (↑1) (HMul.hMul (↑x.2) …
  -/
  rw [OneMemClass.coe_one, one_mul, one_mul]
  -- The hypothesis `a ≠ 0` in `P` is equivalent to this
  have b1ne0 : b.1 ≠ 0 := by
    intro hb1
    have m0 : (LocalizationMap.toMap fl) 0 = 0 := f.map_zero'
    have a0 : a * (LocalizationMap.toMap fl) b.2 = 0 ↔ a = 0 :=
      (f.toLocalizationMap.map_units' b.2).mul_left_eq_zero
    rw [hb1, m0, a0] at hb
    exact ha hb
  have main : g (b.1 * (x.2 * y.1)) = g (b.1 * (y.2 * x.1)) :=
    calc
      g (b.1 * (x.2 * y.1)) = g b.1 * (g x.2 * g y.1) := by rw [map_mul g,map_mul g]
      _ = a * g b.2 * (g x.2 * (w * g y.2)) := by rw [hb, hy]
      _ = a * w * g b.2 * (g x.2 * g y.2) := by
        rw [← mul_assoc, ← mul_assoc _ w, mul_comm _ w, mul_assoc w, mul_assoc,
          ← mul_assoc w, ← mul_assoc w, mul_comm w]
      _ = a * z * g b.2 * (g x.2 * g y.2) := by rw [hazw]
      _ = a * g b.2 * (z * g x.2 * g y.2) := by
        rw [mul_assoc a, mul_comm z, ← mul_assoc a, mul_assoc, mul_assoc z]
      _ = g b.1 * g (y.2 * x.1) := by rw [hx, hb, mul_comm (g x.1), ← map_mul g]
      _ = g (b.1 * (y.2 * x.1)) := by rw [← map_mul g]
 -- The hypothesis `h` gives that `f` (so, `g`) is injective, and we can cancel out `b.1`.
  exact (IsLeftCancelMulZero.mul_left_cancel_of_ne_zero b1ne0
      ((LocalizationMap.toMap_injective_iff fl).mpr h main)).symm


/-- Given a Localization map `f : M →*₀ N` for a Submonoid `S ⊆ M`,
if `M` is a cancellative monoid with zero, and all elements of `S` are
regular, then N is a cancellative monoid with zero. -/
theorem isLeftRegular_of_le_isCancelMulZero (f : LocalizationWithZeroMap S N)
    [IsCancelMulZero M] (h : ∀ ⦃x⦄, x ∈ S → IsRegular x) : IsCancelMulZero N := by
  have : IsLeftCancelMulZero N :=
    leftCancelMulZero_of_le_isLeftRegular f (fun x h' => (h h').left)
  /-
    M : Type u_1
    inst✝² : CommMonoidWithZero M
    S : Submonoid M
    N : Type u_2
    inst✝¹ : CommMonoidWithZero N
    f : S.LocalizationWithZeroMap N
    inst✝ : IsCancelMulZero M
    h : ∀ ⦃x : M⦄, Membership.mem S x → IsRegular x
    this : IsLeftCancelMulZero N
    ⊢ IsCancelMulZero N
  -/
  exact IsLeftCancelMulZero.to_isCancelMulZero
  /-
    🎉 no goals
  -/


