lemma isLocalHom_of_exists_map_ne_one [FunLike F G₀ M] [MonoidHomClass F G₀ M] {f : F}
    (hf : ∃ x : G₀, f x ≠ 1) : IsLocalHom f where
  map_nonunit a h := by
    /-
      M : Type u_1
      G₀ : Type u_3
      F : Type u_6
      inst✝³ : Monoid M
      inst✝² : GroupWithZero G₀
      inst✝¹ : FunLike F G₀ M
      inst✝ : MonoidHomClass F G₀ M
      f : F
      hf : Exists fun x => Ne (f x) 1
      a : G₀
      h : IsUnit (f a)
      ⊢ IsUnit a
    -/
    rcases eq_or_ne a 0 with (rfl | h)
      /-
        case inl
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        hf : Exists fun x => Ne (f x) 1
        h : IsUnit (f 0)
        ⊢ IsUnit 0
      -/
    · obtain ⟨t, ht⟩ := hf
      /-
        case inl.intro
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        h : IsUnit (f 0)
        t : G₀
        ht : Ne (f t) 1
        ⊢ IsUnit 0
      -/
      refine (ht ?_).elim
      /-
        case inl.intro
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        h : IsUnit (f 0)
        t : G₀
        ht : Ne (f t) 1
        ⊢ Eq (f t) 1
      -/
      have := map_mul f t 0
      /-
        case inl.intro
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        h : IsUnit (f 0)
        t : G₀
        ht : Ne (f t) 1
        this : Eq (f (HMul.hMul t 0)) (HMul.hMul (f t) (f 0))
        ⊢ Eq (f t) 1
      -/
      rw [← one_mul (f (t * 0)), mul_zero] at this
      /-
        case inl.intro
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        h : IsUnit (f 0)
        t : G₀
        ht : Ne (f t) 1
        this : Eq (HMul.hMul 1 (f 0)) (HMul.hMul (f t) (f 0))
        ⊢ Eq (f t) 1
      -/
      exact (h.mul_right_cancel this).symm
      /-
        🎉 no goals
      -/
      /-
        case inr
        M : Type u_1
        G₀ : Type u_3
        F : Type u_6
        inst✝³ : Monoid M
        inst✝² : GroupWithZero G₀
        inst✝¹ : FunLike F G₀ M
        inst✝ : MonoidHomClass F G₀ M
        f : F
        hf : Exists fun x => Ne (f x) 1
        a : G₀
        h✝ : IsUnit (f a)
        h : Ne a 0
        ⊢ IsUnit a
      -/
    · exact ⟨⟨a, a⁻¹, mul_inv_cancel₀ h, inv_mul_cancel₀ h⟩, rfl⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_of_exists_map_ne_one := isLocalHom_of_exists_map_ne_one


instance [GroupWithZero G₀] [FunLike F G₀ M₀] [MonoidWithZeroHomClass F G₀ M₀] [Nontrivial M₀]
    (f : F) : IsLocalHom f :=
                                         /-
                                           M : Type u_1
                                           M₀ : Type u_2
                                           G₀ : Type u_3
                                           M₀' : Type u_4
                                           G₀' : Type u_5
                                           F : Type u_6
                                           F' : Type u_7
                                           inst✝⁶ : MonoidWithZero M₀
                                           inst✝⁵ : Monoid M
                                           inst✝⁴ inst✝³ : GroupWithZero G₀
                                           inst✝² : FunLike F G₀ M₀
                                           inst✝¹ : MonoidWithZeroHomClass F G₀ M₀
                                           inst✝ : Nontrivial M₀
                                           f : F
                                           ⊢ Ne (f 0) 1
                                         -/
  isLocalHom_of_exists_map_ne_one ⟨0, by simp⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- The `MonoidWithZero` version of `div_eq_div_iff_mul_eq_mul`. -/
protected lemma div_eq_div_iff (hbd : Commute b d) (hb : b ≠ 0) (hd : d ≠ 0) :
    a / b = c / d ↔ a * d = c * b := hbd.div_eq_div_iff_of_isUnit hb.isUnit hd.isUnit


theorem map_ne_zero : f a ≠ 0 ↔ a ≠ 0 :=
  ⟨fun hfa ha => hfa <| ha.symm ▸ map_zero f, fun ha => ((IsUnit.mk0 a ha).map f).ne_zero⟩


@[simp]
theorem map_eq_zero : f a = 0 ↔ a = 0 :=
  not_iff_not.1 (map_ne_zero f)


theorem eq_on_inv₀ [MonoidWithZeroHomClass F' G₀ M₀'] (f g : F') (h : f a = g a) :
    f a⁻¹ = g a⁻¹ := by
  /-
    G₀ : Type u_3
    M₀' : Type u_4
    F' : Type u_7
    inst✝³ : GroupWithZero G₀
    inst✝² : MonoidWithZero M₀'
    inst✝¹ : FunLike F' G₀ M₀'
    a : G₀
    inst✝ : MonoidWithZeroHomClass F' G₀ M₀'
    f g : F'
    h : Eq (f a) (g a)
    ⊢ Eq (f (Inv.inv a)) (g (Inv.inv a))
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      G₀ : Type u_3
      M₀' : Type u_4
      F' : Type u_7
      inst✝³ : GroupWithZero G₀
      inst✝² : MonoidWithZero M₀'
      inst✝¹ : FunLike F' G₀ M₀'
      inst✝ : MonoidWithZeroHomClass F' G₀ M₀'
      f g : F'
      h : Eq (f 0) (g 0)
      ⊢ Eq (f (Inv.inv 0)) (g (Inv.inv 0))
    -/
  · rw [inv_zero, map_zero, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G₀ : Type u_3
      M₀' : Type u_4
      F' : Type u_7
      inst✝³ : GroupWithZero G₀
      inst✝² : MonoidWithZero M₀'
      inst✝¹ : FunLike F' G₀ M₀'
      a : G₀
      inst✝ : MonoidWithZeroHomClass F' G₀ M₀'
      f g : F'
      h : Eq (f a) (g a)
      ha : Ne a 0
      ⊢ Eq (f (Inv.inv a)) (g (Inv.inv a))
    -/
  · exact (IsUnit.mk0 a ha).eq_on_inv f g h
    /-
      🎉 no goals
    -/


/-- A monoid homomorphism between groups with zeros sending `0` to `0` sends `a⁻¹` to `(f a)⁻¹`. -/
@[simp]
theorem map_inv₀ : f a⁻¹ = (f a)⁻¹ := by
  /-
    G₀ : Type u_3
    G₀' : Type u_5
    F : Type u_6
    inst✝³ : GroupWithZero G₀
    inst✝² : GroupWithZero G₀'
    inst✝¹ : FunLike F G₀ G₀'
    inst✝ : MonoidWithZeroHomClass F G₀ G₀'
    f : F
    a : G₀
    ⊢ Eq (f (Inv.inv a)) (Inv.inv (f a))
  -/
  by_cases h : a = 0
    /-
      case pos
      G₀ : Type u_3
      G₀' : Type u_5
      F : Type u_6
      inst✝³ : GroupWithZero G₀
      inst✝² : GroupWithZero G₀'
      inst✝¹ : FunLike F G₀ G₀'
      inst✝ : MonoidWithZeroHomClass F G₀ G₀'
      f : F
      a : G₀
      h : Eq a 0
      ⊢ Eq (f (Inv.inv a)) (Inv.inv (f a))
    -/
  · simp [h, map_zero f]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G₀ : Type u_3
      G₀' : Type u_5
      F : Type u_6
      inst✝³ : GroupWithZero G₀
      inst✝² : GroupWithZero G₀'
      inst✝¹ : FunLike F G₀ G₀'
      inst✝ : MonoidWithZeroHomClass F G₀ G₀'
      f : F
      a : G₀
      h : Not (Eq a 0)
      ⊢ Eq (f (Inv.inv a)) (Inv.inv (f a))
    -/
  · apply eq_inv_of_mul_eq_one_left
    /-
      case neg.h
      G₀ : Type u_3
      G₀' : Type u_5
      F : Type u_6
      inst✝³ : GroupWithZero G₀
      inst✝² : GroupWithZero G₀'
      inst✝¹ : FunLike F G₀ G₀'
      inst✝ : MonoidWithZeroHomClass F G₀ G₀'
      f : F
      a : G₀
      h : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (f (Inv.inv a)) (f a)) 1
    -/
    rw [← map_mul, inv_mul_cancel₀ h, map_one]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_div₀ : f (a / b) = f a / f b :=
  map_div' f (map_inv₀ f) a b


/-- We define the inverse as a `MonoidWithZeroHom` by extending the inverse map by zero
on non-units. -/
noncomputable def MonoidWithZero.inverse {M : Type*} [CommMonoidWithZero M] :
    M →*₀ M where
  toFun := Ring.inverse
  map_zero' := Ring.inverse_zero _
  map_one' := Ring.inverse_one _
  map_mul' x y := (Ring.mul_inverse_rev x y).trans (mul_comm _ _)


@[simp]
theorem MonoidWithZero.coe_inverse {M : Type*} [CommMonoidWithZero M] :
    (MonoidWithZero.inverse : M → M) = Ring.inverse :=
  rfl


@[simp]
theorem MonoidWithZero.inverse_apply {M : Type*} [CommMonoidWithZero M] (a : M) :
    MonoidWithZero.inverse a = Ring.inverse a :=
  rfl


/-- Inversion on a commutative group with zero, considered as a monoid with zero homomorphism. -/
def invMonoidWithZeroHom {G₀ : Type*} [CommGroupWithZero G₀] : G₀ →*₀ G₀ :=
  { invMonoidHom with map_zero' := inv_zero }


@[simp]
theorem smul_mk0 {α : Type*} [SMul G₀ α] {g : G₀} (hg : g ≠ 0) (a : α) : mk0 g hg • a = g • a :=
  rfl


/-- If a monoid homomorphism `f` between two `GroupWithZero`s maps `0` to `0`, then it maps `x^n`,
`n : ℤ`, to `(f x)^n`. -/
@[simp]
theorem map_zpow₀ {F G₀ G₀' : Type*} [GroupWithZero G₀] [GroupWithZero G₀'] [FunLike F G₀ G₀']
    [MonoidWithZeroHomClass F G₀ G₀'] (f : F) (x : G₀) (n : ℤ) : f (x ^ n) = f x ^ n :=
  map_zpow' f (map_inv₀ f) x n


