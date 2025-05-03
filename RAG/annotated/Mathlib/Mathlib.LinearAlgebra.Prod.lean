/-- The first projection of a product is a linear map. -/
def fst : M × M₂ →ₗ[R] M where
  toFun := Prod.fst
  map_add' _x _y := rfl
  map_smul' _x _y := rfl


/-- The second projection of a product is a linear map. -/
def snd : M × M₂ →ₗ[R] M₂ where
  toFun := Prod.snd
  map_add' _x _y := rfl
  map_smul' _x _y := rfl


@[simp]
theorem fst_apply (x : M × M₂) : fst R M M₂ x = x.1 :=
  rfl


@[simp]
theorem snd_apply (x : M × M₂) : snd R M M₂ x = x.2 :=
  rfl


@[simp, norm_cast] lemma coe_fst : ⇑(fst R M M₂) = Prod.fst := rfl


@[simp, norm_cast] lemma coe_snd : ⇑(snd R M M₂) = Prod.snd := rfl


theorem fst_surjective : Function.Surjective (fst R M M₂) := fun x => ⟨(x, 0), rfl⟩


theorem snd_surjective : Function.Surjective (snd R M M₂) := fun x => ⟨(0, x), rfl⟩


/-- The prod of two linear maps is a linear map. -/
@[simps]
def prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) : M →ₗ[R] M₂ × M₃ where
  toFun := Pi.prod f g
                     /-
                       R : Type u
                       K : Type u'
                       M : Type v
                       V : Type v'
                       M₂ : Type w
                       V₂ : Type w'
                       M₃ : Type y
                       V₃ : Type y'
                       M₄ : Type z
                       ι : Type x
                       M₅ : Type u_1
                       M₆ : Type u_2
                       S : Type u_3
                       inst✝¹³ : Semiring R
                       inst✝¹² : Semiring S
                       inst✝¹¹ : AddCommMonoid M
                       inst✝¹⁰ : AddCommMonoid M₂
                       inst✝⁹ : AddCommMonoid M₃
                       inst✝⁸ : AddCommMonoid M₄
                       inst✝⁷ : AddCommMonoid M₅
                       inst✝⁶ : AddCommMonoid M₆
                       inst✝⁵ : Module R M
                       inst✝⁴ : Module R M₂
                       inst✝³ : Module R M₃
                       inst✝² : Module R M₄
                       inst✝¹ : Module R M₅
                       inst✝ : Module R M₆
                       f✝ f : LinearMap (RingHom.id R) M M₂
                       g : LinearMap (RingHom.id R) M M₃
                       x y : M
                       ⊢ Eq (Pi.prod (⇑f) (⇑g) (HAdd.hAdd x y)) (HAdd.hAdd (Pi.prod (⇑f) (⇑g) x) (Pi. …
                     -/
  map_add' x y := by simp only [Pi.prod, Prod.mk_add_mk, map_add]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u
                        K : Type u'
                        M : Type v
                        V : Type v'
                        M₂ : Type w
                        V₂ : Type w'
                        M₃ : Type y
                        V₃ : Type y'
                        M₄ : Type z
                        ι : Type x
                        M₅ : Type u_1
                        M₆ : Type u_2
                        S : Type u_3
                        inst✝¹³ : Semiring R
                        inst✝¹² : Semiring S
                        inst✝¹¹ : AddCommMonoid M
                        inst✝¹⁰ : AddCommMonoid M₂
                        inst✝⁹ : AddCommMonoid M₃
                        inst✝⁸ : AddCommMonoid M₄
                        inst✝⁷ : AddCommMonoid M₅
                        inst✝⁶ : AddCommMonoid M₆
                        inst✝⁵ : Module R M
                        inst✝⁴ : Module R M₂
                        inst✝³ : Module R M₃
                        inst✝² : Module R M₄
                        inst✝¹ : Module R M₅
                        inst✝ : Module R M₆
                        f✝ f : LinearMap (RingHom.id R) M M₂
                        g : LinearMap (RingHom.id R) M M₃
                        c : R
                        x : M
                        ⊢ Eq ({ toFun := Pi.prod ⇑f ⇑g, map_add' := ⋯ }.toFun (HSMul.hSMul c x)) (HSMu …
                      -/
  map_smul' c x := by simp only [Pi.prod, Prod.smul_mk, map_smul, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/


theorem coe_prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[simp]
theorem fst_prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) : (fst R M₂ M₃).comp (prod f g) = f := rfl


@[simp]
theorem snd_prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) : (snd R M₂ M₃).comp (prod f g) = g := rfl


@[simp]
theorem pair_fst_snd : prod (fst R M M₂) (snd R M M₂) = LinearMap.id := rfl


theorem prod_comp (f : M₂ →ₗ[R] M₃) (g : M₂ →ₗ[R] M₄)
    (h : M →ₗ[R] M₂) : (f.prod g).comp h = (f.comp h).prod (g.comp h) :=
  rfl


/-- Taking the product of two maps with the same domain is equivalent to taking the product of
their codomains.

See note [bundled maps over different rings] for why separate `R` and `S` semirings are used. -/
@[simps]
def prodEquiv [Module S M₂] [Module S M₃] [SMulCommClass R S M₂] [SMulCommClass R S M₃] :
    ((M →ₗ[R] M₂) × (M →ₗ[R] M₃)) ≃ₗ[S] M →ₗ[R] M₂ × M₃ where
  toFun f := f.1.prod f.2
  invFun f := ((fst _ _ _).comp f, (snd _ _ _).comp f)
                   /-
                     R : Type u
                     K : Type u'
                     M : Type v
                     V : Type v'
                     M₂ : Type w
                     V₂ : Type w'
                     M₃ : Type y
                     V₃ : Type y'
                     M₄ : Type z
                     ι : Type x
                     M₅ : Type u_1
                     M₆ : Type u_2
                     S : Type u_3
                     inst✝¹⁷ : Semiring R
                     inst✝¹⁶ : Semiring S
                     inst✝¹⁵ : AddCommMonoid M
                     inst✝¹⁴ : AddCommMonoid M₂
                     inst✝¹³ : AddCommMonoid M₃
                     inst✝¹² : AddCommMonoid M₄
                     inst✝¹¹ : AddCommMonoid M₅
                     inst✝¹⁰ : AddCommMonoid M₆
                     inst✝⁹ : Module R M
                     inst✝⁸ : Module R M₂
                     inst✝⁷ : Module R M₃
                     inst✝⁶ : Module R M₄
                     inst✝⁵ : Module R M₅
                     inst✝⁴ : Module R M₆
                     f✝ : LinearMap (RingHom.id R) M M₂
                     inst✝³ : Module S M₂
                     inst✝² : Module S M₃
                     inst✝¹ : SMulCommClass R S M₂
                     inst✝ : SMulCommClass R S M₃
                     f : Prod (LinearMap (RingHom.id R) M M₂) (LinearMap (RingHom.id R) M M₃)
                     ⊢ Eq ((fun f => { fst := (LinearMap.fst R M₂ M₃).comp f, snd := (LinearMap.snd …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      R : Type u
                      K : Type u'
                      M : Type v
                      V : Type v'
                      M₂ : Type w
                      V₂ : Type w'
                      M₃ : Type y
                      V₃ : Type y'
                      M₄ : Type z
                      ι : Type x
                      M₅ : Type u_1
                      M₆ : Type u_2
                      S : Type u_3
                      inst✝¹⁷ : Semiring R
                      inst✝¹⁶ : Semiring S
                      inst✝¹⁵ : AddCommMonoid M
                      inst✝¹⁴ : AddCommMonoid M₂
                      inst✝¹³ : AddCommMonoid M₃
                      inst✝¹² : AddCommMonoid M₄
                      inst✝¹¹ : AddCommMonoid M₅
                      inst✝¹⁰ : AddCommMonoid M₆
                      inst✝⁹ : Module R M
                      inst✝⁸ : Module R M₂
                      inst✝⁷ : Module R M₃
                      inst✝⁶ : Module R M₄
                      inst✝⁵ : Module R M₅
                      inst✝⁴ : Module R M₆
                      f✝ : LinearMap (RingHom.id R) M M₂
                      inst✝³ : Module S M₂
                      inst✝² : Module S M₃
                      inst✝¹ : SMulCommClass R S M₂
                      inst✝ : SMulCommClass R S M₃
                      f : LinearMap (RingHom.id R) M (Prod M₂ M₃)
                      ⊢ Eq ({ toFun := fun f => f.1.prod f.2, map_add' := ⋯, map_smul' := ⋯ }.toFun  …
                    -/
                            /-
                              🎉 no goals
                            -/
  right_inv f := by ext <;> rfl
                            /-
                              🎉 no goals
                            -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- The left injection into a product is a linear map. -/
def inl : M →ₗ[R] M × M₂ :=
  prod LinearMap.id 0


/-- The right injection into a product is a linear map. -/
def inr : M₂ →ₗ[R] M × M₂ :=
  prod 0 LinearMap.id


theorem range_inl : range (inl R M M₂) = ker (snd R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.ker (LinearMap.snd R  …
  -/
  ext x
  /-
    case h
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    x : Prod M M₂
    ⊢ Iff (Membership.mem (LinearMap.range (LinearMap.inl R M M₂)) x) (Membership. …
  -/
  simp only [mem_ker, mem_range]
  /-
    case h
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    x : Prod M M₂
    ⊢ Iff (Exists fun y => Eq ((LinearMap.inl R M M₂) y) x) (Eq ((LinearMap.snd R  …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      ⊢ (Exists fun y => Eq ((LinearMap.inl R M M₂) y) x) → Eq ((LinearMap.snd R M M …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mp.intro
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      y : M
      ⊢ Eq ((LinearMap.snd R M M₂) ((LinearMap.inl R M M₂) y)) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      ⊢ Eq ((LinearMap.snd R M M₂) x) 0 → Exists fun y => Eq ((LinearMap.inl R M M₂) …
    -/
  · intro h
    /-
      case h.mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      h : Eq ((LinearMap.snd R M M₂) x) 0
      ⊢ Exists fun y => Eq ((LinearMap.inl R M M₂) y) x
    -/
    exact ⟨x.fst, Prod.ext rfl h.symm⟩
    /-
      🎉 no goals
    -/


theorem ker_snd : ker (snd R M M₂) = range (inl R M M₂) :=
  Eq.symm <| range_inl R M M₂


theorem range_inr : range (inr R M M₂) = ker (fst R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (LinearMap.range (LinearMap.inr R M M₂)) (LinearMap.ker (LinearMap.fst R  …
  -/
  ext x
  /-
    case h
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    x : Prod M M₂
    ⊢ Iff (Membership.mem (LinearMap.range (LinearMap.inr R M M₂)) x) (Membership. …
  -/
  simp only [mem_ker, mem_range]
  /-
    case h
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    x : Prod M M₂
    ⊢ Iff (Exists fun y => Eq ((LinearMap.inr R M M₂) y) x) (Eq ((LinearMap.fst R  …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      ⊢ (Exists fun y => Eq ((LinearMap.inr R M M₂) y) x) → Eq ((LinearMap.fst R M M …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mp.intro
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      y : M₂
      ⊢ Eq ((LinearMap.fst R M M₂) ((LinearMap.inr R M M₂) y)) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      ⊢ Eq ((LinearMap.fst R M M₂) x) 0 → Exists fun y => Eq ((LinearMap.inr R M M₂) …
    -/
  · intro h
    /-
      case h.mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : Prod M M₂
      h : Eq ((LinearMap.fst R M M₂) x) 0
      ⊢ Exists fun y => Eq ((LinearMap.inr R M M₂) y) x
    -/
    exact ⟨x.snd, Prod.ext h.symm rfl⟩
    /-
      🎉 no goals
    -/


theorem ker_fst : ker (fst R M M₂) = range (inr R M M₂) :=
  Eq.symm <| range_inr R M M₂


@[simp] theorem fst_comp_inl : fst R M M₂ ∘ₗ inl R M M₂ = id := rfl


@[simp] theorem snd_comp_inl : snd R M M₂ ∘ₗ inl R M M₂ = 0 := rfl


@[simp] theorem fst_comp_inr : fst R M M₂ ∘ₗ inr R M M₂ = 0 := rfl


@[simp] theorem snd_comp_inr : snd R M M₂ ∘ₗ inr R M M₂ = id := rfl


@[simp]
theorem coe_inl : (inl R M M₂ : M → M × M₂) = fun x => (x, 0) :=
  rfl


theorem inl_apply (x : M) : inl R M M₂ x = (x, 0) :=
  rfl


@[simp]
theorem coe_inr : (inr R M M₂ : M₂ → M × M₂) = Prod.mk 0 :=
  rfl


theorem inr_apply (x : M₂) : inr R M M₂ x = (0, x) :=
  rfl


theorem inl_eq_prod : inl R M M₂ = prod LinearMap.id 0 :=
  rfl


theorem inr_eq_prod : inr R M M₂ = prod 0 LinearMap.id :=
  rfl


                                                                       /-
                                                                         R : Type u
                                                                         M : Type v
                                                                         M₂ : Type w
                                                                         inst✝⁴ : Semiring R
                                                                         inst✝³ : AddCommMonoid M
                                                                         inst✝² : AddCommMonoid M₂
                                                                         inst✝¹ : Module R M
                                                                         inst✝ : Module R M₂
                                                                         x✝ : M
                                                                         ⊢ ∀ ⦃a₂ : M⦄, Eq ((LinearMap.inl R M M₂) x✝) ((LinearMap.inl R M M₂) a₂) → Eq  …
                                                                       -/
theorem inl_injective : Function.Injective (inl R M M₂) := fun _ => by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                       /-
                                                                         R : Type u
                                                                         M : Type v
                                                                         M₂ : Type w
                                                                         inst✝⁴ : Semiring R
                                                                         inst✝³ : AddCommMonoid M
                                                                         inst✝² : AddCommMonoid M₂
                                                                         inst✝¹ : Module R M
                                                                         inst✝ : Module R M₂
                                                                         x✝ : M₂
                                                                         ⊢ ∀ ⦃a₂ : M₂⦄, Eq ((LinearMap.inr R M M₂) x✝) ((LinearMap.inr R M M₂) a₂) → Eq …
                                                                       -/
theorem inr_injective : Function.Injective (inr R M M₂) := fun _ => by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- The coprod function `x : M × M₂ ↦ f x.1 + g x.2` is a linear map. -/
def coprod (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) : M × M₂ →ₗ[R] M₃ :=
  f.comp (fst _ _ _) + g.comp (snd _ _ _)


@[simp]
theorem coprod_apply (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) (x : M × M₂) :
    coprod f g x = f x.1 + g x.2 :=
  rfl


@[simp]
theorem coprod_inl (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) : (coprod f g).comp (inl R M M₂) = f := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    ⊢ Eq ((f.coprod g).comp (LinearMap.inl R M M₂)) f
  -/
  ext; simp only [map_zero, add_zero, coprod_apply, inl_apply, comp_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem coprod_inr (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) : (coprod f g).comp (inr R M M₂) = g := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    ⊢ Eq ((f.coprod g).comp (LinearMap.inr R M M₂)) g
  -/
  ext; simp only [map_zero, coprod_apply, inr_apply, zero_add, comp_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem coprod_inl_inr : coprod (inl R M M₂) (inr R M M₂) = LinearMap.id := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq ((LinearMap.inl R M M₂).coprod (LinearMap.inr R M M₂)) LinearMap.id
  -/
  ext <;>
    /-
      case h.fst
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x✝ : Prod M M₂
      ⊢ Eq (((LinearMap.inl R M M₂).coprod (LinearMap.inr R M M₂)) x✝).1 (LinearMap. …
    -/
    /-
      🎉 no goals
    -/
    simp only [Prod.mk_add_mk, add_zero, id_apply, coprod_apply, inl_apply, inr_apply, zero_add]
    /-
      🎉 no goals
    -/


theorem coprod_zero_left (g : M₂ →ₗ[R] M₃) : (0 : M →ₗ[R] M₃).coprod g = g.comp (snd R M M₂) :=
  zero_add _


theorem coprod_zero_right (f : M →ₗ[R] M₃) : f.coprod (0 : M₂ →ₗ[R] M₃) = f.comp (fst R M M₂) :=
  add_zero _


theorem comp_coprod (f : M₃ →ₗ[R] M₄) (g₁ : M →ₗ[R] M₃) (g₂ : M₂ →ₗ[R] M₃) :
    f.comp (g₁.coprod g₂) = (f.comp g₁).coprod (f.comp g₂) :=
  ext fun x => f.map_add (g₁ x.1) (g₂ x.2)


                                                                 /-
                                                                   R : Type u
                                                                   M : Type v
                                                                   M₂ : Type w
                                                                   inst✝⁴ : Semiring R
                                                                   inst✝³ : AddCommMonoid M
                                                                   inst✝² : AddCommMonoid M₂
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : Module R M₂
                                                                   ⊢ Eq (LinearMap.fst R M M₂) (LinearMap.id.coprod 0)
                                                                 -/
theorem fst_eq_coprod : fst R M M₂ = coprod LinearMap.id 0 := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                                 /-
                                                                   R : Type u
                                                                   M : Type v
                                                                   M₂ : Type w
                                                                   inst✝⁴ : Semiring R
                                                                   inst✝³ : AddCommMonoid M
                                                                   inst✝² : AddCommMonoid M₂
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : Module R M₂
                                                                   ⊢ Eq (LinearMap.snd R M M₂) (LinearMap.coprod 0 LinearMap.id)
                                                                 -/
theorem snd_eq_coprod : snd R M M₂ = coprod 0 LinearMap.id := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem coprod_comp_prod (f : M₂ →ₗ[R] M₄) (g : M₃ →ₗ[R] M₄) (f' : M →ₗ[R] M₂) (g' : M →ₗ[R] M₃) :
    (f.coprod g).comp (f'.prod g') = f.comp f' + g.comp g' :=
  rfl


@[simp]
theorem coprod_map_prod (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) (S : Submodule R M)
    (S' : Submodule R M₂) : (Submodule.prod S S').map (LinearMap.coprod f g) = S.map f ⊔ S'.map g :=
  SetLike.coe_injective <| by
    /-
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      S : Submodule R M
      S' : Submodule R M₂
      ⊢ Eq ↑(Submodule.map (f.coprod g) (S.prod S')) ↑(Max.max (Submodule.map f S) ( …
    -/
    simp only [LinearMap.coprod_apply, Submodule.coe_sup, Submodule.map_coe]
    /-
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      S : Submodule R M
      S' : Submodule R M₂
      ⊢ Eq (Set.image (fun a => HAdd.hAdd (f a.1) (g a.2)) ↑(S.prod S')) (HAdd.hAdd  …
    -/
    rw [← Set.image2_add, Set.image2_image_left, Set.image2_image_right]
    /-
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      S : Submodule R M
      S' : Submodule R M₂
      ⊢ Eq (Set.image (fun a => HAdd.hAdd (f a.1) (g a.2)) ↑(S.prod S')) (Set.image2 …
    -/
    exact Set.image_prod fun m m₂ => f m + g m₂
    /-
      🎉 no goals
    -/


/-- Taking the product of two maps with the same codomain is equivalent to taking the product of
their domains.

See note [bundled maps over different rings] for why separate `R` and `S` semirings are used. -/
@[simps]
def coprodEquiv [Module S M₃] [SMulCommClass R S M₃] :
    ((M →ₗ[R] M₃) × (M₂ →ₗ[R] M₃)) ≃ₗ[S] M × M₂ →ₗ[R] M₃ where
  toFun f := f.1.coprod f.2
  invFun f := (f.comp (inl _ _ _), f.comp (inr _ _ _))
                   /-
                     R : Type u
                     K : Type u'
                     M : Type v
                     V : Type v'
                     M₂ : Type w
                     V₂ : Type w'
                     M₃ : Type y
                     V₃ : Type y'
                     M₄ : Type z
                     ι : Type x
                     M₅ : Type u_1
                     M₆ : Type u_2
                     S : Type u_3
                     inst✝¹⁵ : Semiring R
                     inst✝¹⁴ : Semiring S
                     inst✝¹³ : AddCommMonoid M
                     inst✝¹² : AddCommMonoid M₂
                     inst✝¹¹ : AddCommMonoid M₃
                     inst✝¹⁰ : AddCommMonoid M₄
                     inst✝⁹ : AddCommMonoid M₅
                     inst✝⁸ : AddCommMonoid M₆
                     inst✝⁷ : Module R M
                     inst✝⁶ : Module R M₂
                     inst✝⁵ : Module R M₃
                     inst✝⁴ : Module R M₄
                     inst✝³ : Module R M₅
                     inst✝² : Module R M₆
                     f✝ : LinearMap (RingHom.id R) M M₂
                     inst✝¹ : Module S M₃
                     inst✝ : SMulCommClass R S M₃
                     f : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
                     ⊢ Eq ((fun f => { fst := f.comp (LinearMap.inl R M M₂), snd := f.comp (LinearM …
                   -/
  left_inv f := by simp only [coprod_inl, coprod_inr]
                   /-
                     🎉 no goals
                   -/
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      S : Type u_3
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : Semiring S
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M₄
      inst✝⁹ : AddCommMonoid M₅
      inst✝⁸ : AddCommMonoid M₆
      inst✝⁷ : Module R M
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module R M₃
      inst✝⁴ : Module R M₄
      inst✝³ : Module R M₅
      inst✝² : Module R M₆
      f : LinearMap (RingHom.id R) M M₂
      inst✝¹ : Module S M₃
      inst✝ : SMulCommClass R S M₃
      a b : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
      ⊢ Eq ((fun f => f.1.coprod f.2) (HAdd.hAdd a b)) (HAdd.hAdd ((fun f => f.1.cop …
    -/
                    /-
                      R : Type u
                      K : Type u'
                      M : Type v
                      V : Type v'
                      M₂ : Type w
                      V₂ : Type w'
                      M₃ : Type y
                      V₃ : Type y'
                      M₄ : Type z
                      ι : Type x
                      M₅ : Type u_1
                      M₆ : Type u_2
                      S : Type u_3
                      inst✝¹⁵ : Semiring R
                      inst✝¹⁴ : Semiring S
                      inst✝¹³ : AddCommMonoid M
                      inst✝¹² : AddCommMonoid M₂
                      inst✝¹¹ : AddCommMonoid M₃
                      inst✝¹⁰ : AddCommMonoid M₄
                      inst✝⁹ : AddCommMonoid M₅
                      inst✝⁸ : AddCommMonoid M₆
                      inst✝⁷ : Module R M
                      inst✝⁶ : Module R M₂
                      inst✝⁵ : Module R M₃
                      inst✝⁴ : Module R M₄
                      inst✝³ : Module R M₅
                      inst✝² : Module R M₆
                      f✝ : LinearMap (RingHom.id R) M M₂
                      inst✝¹ : Module S M₃
                      inst✝ : SMulCommClass R S M₃
                      f : LinearMap (RingHom.id R) (Prod M M₂) M₃
                      ⊢ Eq ({ toFun := fun f => f.1.coprod f.2, map_add' := ⋯, map_smul' := ⋯ }.toFu …
                    -/
    /-
      case h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      S : Type u_3
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : Semiring S
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M₄
      inst✝⁹ : AddCommMonoid M₅
      inst✝⁸ : AddCommMonoid M₆
      inst✝⁷ : Module R M
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module R M₃
      inst✝⁴ : Module R M₄
      inst✝³ : Module R M₅
      inst✝² : Module R M₆
      f : LinearMap (RingHom.id R) M M₂
      inst✝¹ : Module S M₃
      inst✝ : SMulCommClass R S M₃
      a b : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
      x✝ : Prod M M₂
      ⊢ Eq (((fun f => f.1.coprod f.2) (HAdd.hAdd a b)) x✝) ((HAdd.hAdd ((fun f => f …
    -/
  right_inv f := by simp only [← comp_coprod, comp_id, coprod_inl_inr]
    /-
      🎉 no goals
    -/
                    /-
                      🎉 no goals
                    -/
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      S : Type u_3
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : Semiring S
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M₄
      inst✝⁹ : AddCommMonoid M₅
      inst✝⁸ : AddCommMonoid M₆
      inst✝⁷ : Module R M
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module R M₃
      inst✝⁴ : Module R M₄
      inst✝³ : Module R M₅
      inst✝² : Module R M₆
      f : LinearMap (RingHom.id R) M M₂
      inst✝¹ : Module S M₃
      inst✝ : SMulCommClass R S M₃
      r : S
      a : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
      ⊢ Eq ({ toFun := fun f => f.1.coprod f.2, map_add' := ⋯ }.toFun (HSMul.hSMul r …
    -/
  map_add' a b := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      S : Type u_3
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : Semiring S
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M₄
      inst✝⁹ : AddCommMonoid M₅
      inst✝⁸ : AddCommMonoid M₆
      inst✝⁷ : Module R M
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module R M₃
      inst✝⁴ : Module R M₄
      inst✝³ : Module R M₅
      inst✝² : Module R M₆
      f : LinearMap (RingHom.id R) M M₂
      inst✝¹ : Module S M₃
      inst✝ : SMulCommClass R S M₃
      r : S
      a : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
      ⊢ Eq ((HSMul.hSMul r a.1).coprod (HSMul.hSMul r a.2)) (HSMul.hSMul r (a.1.copr …
    -/
    ext
    /-
      case h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      S : Type u_3
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : Semiring S
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M₄
      inst✝⁹ : AddCommMonoid M₅
      inst✝⁸ : AddCommMonoid M₆
      inst✝⁷ : Module R M
      inst✝⁶ : Module R M₂
      inst✝⁵ : Module R M₃
      inst✝⁴ : Module R M₄
      inst✝³ : Module R M₅
      inst✝² : Module R M₆
      f : LinearMap (RingHom.id R) M M₂
      inst✝¹ : Module S M₃
      inst✝ : SMulCommClass R S M₃
      r : S
      a : Prod (LinearMap (RingHom.id R) M M₃) (LinearMap (RingHom.id R) M₂ M₃)
      x✝ : Prod M M₂
      ⊢ Eq (((HSMul.hSMul r a.1).coprod (HSMul.hSMul r a.2)) x✝) ((HSMul.hSMul r (a. …
    -/
    simp only [Prod.snd_add, add_apply, coprod_apply, Prod.fst_add, add_add_add_comm]
    /-
      🎉 no goals
    -/
  map_smul' r a := by
    dsimp
    ext
    simp only [smul_add, smul_apply, Prod.smul_snd, Prod.smul_fst, coprod_apply]


theorem prod_ext_iff {f g : M × M₂ →ₗ[R] M₃} :
    f = g ↔ f.comp (inl _ _ _) = g.comp (inl _ _ _) ∧ f.comp (inr _ _ _) = g.comp (inr _ _ _) :=
  (coprodEquiv ℕ).symm.injective.eq_iff.symm.trans Prod.ext_iff


/--
Split equality of linear maps from a product into linear maps over each component, to allow `ext`
to apply lemmas specific to `M →ₗ M₃` and `M₂ →ₗ M₃`.

See note [partially-applied ext lemmas]. -/
@[ext 1100]
theorem prod_ext {f g : M × M₂ →ₗ[R] M₃} (hl : f.comp (inl _ _ _) = g.comp (inl _ _ _))
    (hr : f.comp (inr _ _ _) = g.comp (inr _ _ _)) : f = g :=
  prod_ext_iff.2 ⟨hl, hr⟩


/-- `prod.map` of two linear maps. -/
def prodMap (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₄) : M × M₂ →ₗ[R] M₃ × M₄ :=
  (f.comp (fst R M M₂)).prod (g.comp (snd R M M₂))


theorem coe_prodMap (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₄) : ⇑(f.prodMap g) = Prod.map f g :=
  rfl


@[simp]
theorem prodMap_apply (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₄) (x) : f.prodMap g x = (f x.1, g x.2) :=
  rfl


theorem prodMap_comap_prod (f : M →ₗ[R] M₂) (g : M₃ →ₗ[R] M₄) (S : Submodule R M₂)
    (S' : Submodule R M₄) :
    (Submodule.prod S S').comap (LinearMap.prodMap f g) = (S.comap f).prod (S'.comap g) :=
  SetLike.coe_injective <| Set.preimage_prod_map_prod f g _ _


theorem ker_prodMap (f : M →ₗ[R] M₂) (g : M₃ →ₗ[R] M₄) :
    ker (LinearMap.prodMap f g) = Submodule.prod (ker f) (ker g) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    M₄ : Type z
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : AddCommMonoid M₄
    inst✝³ : Module R M
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Module R M₄
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M₃ M₄
    ⊢ Eq (LinearMap.ker (f.prodMap g)) ((LinearMap.ker f).prod (LinearMap.ker g))
  -/
  dsimp only [ker]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    M₄ : Type z
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : AddCommMonoid M₄
    inst✝³ : Module R M
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Module R M₄
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M₃ M₄
    ⊢ Eq (Submodule.comap (f.prodMap g) Bot.bot) ((Submodule.comap f Bot.bot).prod …
  -/
  rw [← prodMap_comap_prod, Submodule.prod_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem prodMap_id : (id : M →ₗ[R] M).prodMap (id : M₂ →ₗ[R] M₂) = id :=
  rfl


@[simp]
theorem prodMap_one : (1 : M →ₗ[R] M).prodMap (1 : M₂ →ₗ[R] M₂) = 1 :=
  rfl


theorem prodMap_comp (f₁₂ : M →ₗ[R] M₂) (f₂₃ : M₂ →ₗ[R] M₃) (g₁₂ : M₄ →ₗ[R] M₅)
    (g₂₃ : M₅ →ₗ[R] M₆) :
    f₂₃.prodMap g₂₃ ∘ₗ f₁₂.prodMap g₁₂ = (f₂₃ ∘ₗ f₁₂).prodMap (g₂₃ ∘ₗ g₁₂) :=
  rfl


theorem prodMap_mul (f₁₂ : M →ₗ[R] M) (f₂₃ : M →ₗ[R] M) (g₁₂ : M₂ →ₗ[R] M₂) (g₂₃ : M₂ →ₗ[R] M₂) :
    f₂₃.prodMap g₂₃ * f₁₂.prodMap g₁₂ = (f₂₃ * f₁₂).prodMap (g₂₃ * g₁₂) :=
  rfl


theorem prodMap_add (f₁ : M →ₗ[R] M₃) (f₂ : M →ₗ[R] M₃) (g₁ : M₂ →ₗ[R] M₄) (g₂ : M₂ →ₗ[R] M₄) :
    (f₁ + f₂).prodMap (g₁ + g₂) = f₁.prodMap g₁ + f₂.prodMap g₂ :=
  rfl


@[simp]
theorem prodMap_zero : (0 : M →ₗ[R] M₂).prodMap (0 : M₃ →ₗ[R] M₄) = 0 :=
  rfl


@[simp]
theorem prodMap_smul [Module S M₃] [Module S M₄] [SMulCommClass R S M₃] [SMulCommClass R S M₄]
    (s : S) (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₄) : prodMap (s • f) (s • g) = s • prodMap f g :=
  rfl


/-- `LinearMap.prodMap` as a `LinearMap` -/
@[simps]
def prodMapLinear [Module S M₃] [Module S M₄] [SMulCommClass R S M₃] [SMulCommClass R S M₄] :
    (M →ₗ[R] M₃) × (M₂ →ₗ[R] M₄) →ₗ[S] M × M₂ →ₗ[R] M₃ × M₄ where
  toFun f := prodMap f.1 f.2
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- `LinearMap.prodMap` as a `RingHom` -/
@[simps]
def prodMapRingHom : (M →ₗ[R] M) × (M₂ →ₗ[R] M₂) →+* M × M₂ →ₗ[R] M × M₂ where
  toFun f := prodMap f.1 f.2
  map_one' := prodMap_one
  map_zero' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


theorem inl_map_mul (a₁ a₂ : A) :
    LinearMap.inl R A B (a₁ * a₂) = LinearMap.inl R A B a₁ * LinearMap.inl R A B a₂ :=
                   /-
                     R : Type u
                     inst✝⁴ : Semiring R
                     A : Type u_4
                     inst✝³ : NonUnitalNonAssocSemiring A
                     inst✝² : Module R A
                     B : Type u_5
                     inst✝¹ : NonUnitalNonAssocSemiring B
                     inst✝ : Module R B
                     a₁ a₂ : A
                     ⊢ Eq ((LinearMap.inl R A B) (HMul.hMul a₁ a₂)).2 (HMul.hMul ((LinearMap.inl R  …
                   -/
  Prod.ext rfl (by simp)
                   /-
                     🎉 no goals
                   -/


theorem inr_map_mul (b₁ b₂ : B) :
    LinearMap.inr R A B (b₁ * b₂) = LinearMap.inr R A B b₁ * LinearMap.inr R A B b₂ :=
               /-
                 R : Type u
                 inst✝⁴ : Semiring R
                 A : Type u_4
                 inst✝³ : NonUnitalNonAssocSemiring A
                 inst✝² : Module R A
                 B : Type u_5
                 inst✝¹ : NonUnitalNonAssocSemiring B
                 inst✝ : Module R B
                 b₁ b₂ : B
                 ⊢ Eq ((LinearMap.inr R A B) (HMul.hMul b₁ b₂)).1 (HMul.hMul ((LinearMap.inr R  …
               -/
  Prod.ext (by simp) rfl
               /-
                 🎉 no goals
               -/


/-- `LinearMap.prodMap` as an `AlgHom` -/
@[simps!]
def prodMapAlgHom : Module.End R M × Module.End R M₂ →ₐ[R] Module.End R (M × M₂) :=
  { prodMapRingHom R M M₂ with commutes' := fun _ => rfl }


theorem range_coprod (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) : range (f.coprod g) = range f ⊔ range g :=
                            /-
                              R : Type u
                              M : Type v
                              M₂ : Type w
                              M₃ : Type y
                              inst✝⁶ : Semiring R
                              inst✝⁵ : AddCommMonoid M
                              inst✝⁴ : AddCommMonoid M₂
                              inst✝³ : AddCommMonoid M₃
                              inst✝² : Module R M
                              inst✝¹ : Module R M₂
                              inst✝ : Module R M₃
                              f : LinearMap (RingHom.id R) M M₃
                              g : LinearMap (RingHom.id R) M₂ M₃
                              x : M₃
                              ⊢ Iff (Membership.mem (LinearMap.range (f.coprod g)) x) (Membership.mem (Max.m …
                            -/
  Submodule.ext fun x => by simp [mem_sup]
                            /-
                              🎉 no goals
                            -/


theorem isCompl_range_inl_inr : IsCompl (range <| inl R M M₂) (range <| inr R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ IsCompl (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.range (LinearMap …
  -/
  constructor
    /-
      case disjoint
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      ⊢ Disjoint (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.range (LinearMa …
    -/
  · rw [disjoint_def]
    /-
      case disjoint
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      ⊢ ∀ (x : Prod M M₂), Membership.mem (LinearMap.range (LinearMap.inl R M M₂)) x …
    -/
    rintro ⟨_, _⟩ ⟨x, hx⟩ ⟨y, hy⟩
    /-
      case disjoint.mk.intro.intro
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      fst✝ : M
      snd✝ : M₂
      x : M
      hx : Eq ((LinearMap.inl R M M₂) x) { fst := fst✝, snd := snd✝ }
      y : M₂
      hy : Eq ((LinearMap.inr R M M₂) y) { fst := fst✝, snd := snd✝ }
      ⊢ Eq { fst := fst✝, snd := snd✝ } 0
    -/
    simp only [Prod.ext_iff, inl_apply, inr_apply, mem_bot] at hx hy ⊢
    /-
      case disjoint.mk.intro.intro
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      fst✝ : M
      snd✝ : M₂
      x : M
      y : M₂
      hx : And (Eq x fst✝) (Eq 0 snd✝)
      hy : And (Eq 0 fst✝) (Eq y snd✝)
      ⊢ And (Eq fst✝ 0.1) (Eq snd✝ 0.2)
    -/
    exact ⟨hy.1.symm, hx.2.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case codisjoint
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      ⊢ Codisjoint (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.range (Linear …
    -/
  · rw [codisjoint_iff_le_sup]
    /-
      case codisjoint
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      ⊢ LE.le Top.top (Max.max (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.r …
    -/
    rintro ⟨x, y⟩ -
    /-
      case codisjoint.mk
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : M
      y : M₂
      ⊢ Membership.mem (Max.max (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap. …
    -/
    simp only [mem_sup, mem_range, exists_prop]
    /-
      case codisjoint.mk
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : M
      y : M₂
      ⊢ Exists fun y_1 => And (Exists fun y => Eq ((LinearMap.inl R M M₂) y) y_1) (E …
    -/
    refine ⟨(x, 0), ⟨x, rfl⟩, (0, y), ⟨y, rfl⟩, ?_⟩
    /-
      case codisjoint.mk
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      x : M
      y : M₂
      ⊢ Eq (HAdd.hAdd { fst := x, snd := 0 } { fst := 0, snd := y }) { fst := x, snd …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem sup_range_inl_inr : (range <| inl R M M₂) ⊔ (range <| inr R M M₂) = ⊤ :=
  IsCompl.sup_eq_top isCompl_range_inl_inr


theorem disjoint_inl_inr : Disjoint (range <| inl R M M₂) (range <| inr R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Disjoint (LinearMap.range (LinearMap.inl R M M₂)) (LinearMap.range (LinearMa …
  -/
  simp +contextual [disjoint_def, @eq_comm M 0, @eq_comm M₂ 0]
  /-
    🎉 no goals
  -/


theorem map_coprod_prod (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) (p : Submodule R M)
    (q : Submodule R M₂) : map (coprod f g) (p.prod q) = map f p ⊔ map g q := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    p : Submodule R M
    q : Submodule R M₂
    ⊢ Eq (Submodule.map (f.coprod g) (p.prod q)) (Max.max (Submodule.map f p) (Sub …
  -/
  refine le_antisymm ?_ (sup_le (map_le_iff_le_comap.2 ?_) (map_le_iff_le_comap.2 ?_))
    /-
      case refine_1
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      p : Submodule R M
      q : Submodule R M₂
      ⊢ LE.le (Submodule.map (f.coprod g) (p.prod q)) (Max.max (Submodule.map f p) ( …
    -/
  · rw [SetLike.le_def]
    /-
      case refine_1
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      p : Submodule R M
      q : Submodule R M₂
      ⊢ ∀ ⦃x : M₃⦄, Membership.mem (Submodule.map (f.coprod g) (p.prod q)) x → Membe …
    -/
    rintro _ ⟨x, ⟨h₁, h₂⟩, rfl⟩
    /-
      case refine_1.intro.intro.intro
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      p : Submodule R M
      q : Submodule R M₂
      x : Prod M M₂
      h₁ : Membership.mem (↑p) x.1
      h₂ : Membership.mem (↑q) x.2
      ⊢ Membership.mem (Max.max (Submodule.map f p) (Submodule.map g q)) ((f.coprod  …
    -/
    exact mem_sup.2 ⟨_, ⟨_, h₁, rfl⟩, _, ⟨_, h₂, rfl⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      p : Submodule R M
      q : Submodule R M₂
      ⊢ LE.le p (Submodule.comap f (Submodule.map (f.coprod g) (p.prod q)))
    -/
  · exact fun x hx => ⟨(x, 0), by simp [hx]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₃
      g : LinearMap (RingHom.id R) M₂ M₃
      p : Submodule R M
      q : Submodule R M₂
      ⊢ LE.le q (Submodule.comap g (Submodule.map (f.coprod g) (p.prod q)))
    -/
  · exact fun x hx => ⟨(0, x), by simp [hx]⟩
    /-
      🎉 no goals
    -/


theorem comap_prod_prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) (p : Submodule R M₂)
    (q : Submodule R M₃) : comap (prod f g) (p.prod q) = comap f p ⊓ comap g q :=
  Submodule.ext fun _x => Iff.rfl


theorem prod_eq_inf_comap (p : Submodule R M) (q : Submodule R M₂) :
    p.prod q = p.comap (LinearMap.fst R M M₂) ⊓ q.comap (LinearMap.snd R M M₂) :=
  Submodule.ext fun _x => Iff.rfl


theorem prod_eq_sup_map (p : Submodule R M) (q : Submodule R M₂) :
    p.prod q = p.map (LinearMap.inl R M M₂) ⊔ q.map (LinearMap.inr R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p : Submodule R M
    q : Submodule R M₂
    ⊢ Eq (p.prod q) (Max.max (Submodule.map (LinearMap.inl R M M₂) p) (Submodule.m …
  -/
  rw [← map_coprod_prod, coprod_inl_inr, map_id]
  /-
    🎉 no goals
  -/


theorem span_inl_union_inr {s : Set M} {t : Set M₂} :
    span R (inl R M M₂ '' s ∪ inr R M M₂ '' t) = (span R s).prod (span R t) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    s : Set M
    t : Set M₂
    ⊢ Eq (Submodule.span R (Union.union (Set.image (⇑(LinearMap.inl R M M₂)) s) (S …
  -/
  rw [span_union, prod_eq_sup_map, ← span_image, ← span_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_prod (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) : ker (prod f g) = ker f ⊓ ker g := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    ⊢ Eq (LinearMap.ker (f.prod g)) (Min.min (LinearMap.ker f) (LinearMap.ker g))
  -/
  rw [ker, ← prod_bot, comap_prod_prod]; rfl
                                         /-
                                           🎉 no goals
                                         -/


theorem range_prod_le (f : M →ₗ[R] M₂) (g : M →ₗ[R] M₃) :
    range (prod f g) ≤ (range f).prod (range g) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    ⊢ LE.le (LinearMap.range (f.prod g)) ((LinearMap.range f).prod (LinearMap.rang …
  -/
  simp only [SetLike.le_def, prod_apply, mem_range, SetLike.mem_coe, mem_prod, exists_imp]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    ⊢ ∀ ⦃x : Prod M₂ M₃⦄ (x_1 : M), Eq (Pi.prod (⇑f) (⇑g) x_1) x → And (Exists fun …
  -/
  rintro _ x rfl
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    x : M
    ⊢ And (Exists fun y => Eq (f y) (Pi.prod (⇑f) (⇑g) x).1) (Exists fun y => Eq ( …
  -/
  exact ⟨⟨x, rfl⟩, ⟨x, rfl⟩⟩
  /-
    🎉 no goals
  -/


theorem ker_prod_ker_le_ker_coprod {M₂ : Type*} [AddCommMonoid M₂] [Module R M₂] {M₃ : Type*}
    [AddCommMonoid M₃] [Module R M₃] (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃) :
    (ker f).prod (ker g) ≤ ker (f.coprod g) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    ⊢ LE.le ((LinearMap.ker f).prod (LinearMap.ker g)) (LinearMap.ker (f.coprod g))
  -/
  rintro ⟨y, z⟩
  /-
    case mk
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    y : M
    z : M₂
    ⊢ Membership.mem ((LinearMap.ker f).prod (LinearMap.ker g)) { fst := y, snd := …
  -/
  simp +contextual
  /-
    🎉 no goals
  -/


theorem ker_coprod_of_disjoint_range {M₂ : Type*} [AddCommGroup M₂] [Module R M₂] {M₃ : Type*}
    [AddCommGroup M₃] [Module R M₃] (f : M →ₗ[R] M₃) (g : M₂ →ₗ[R] M₃)
    (hd : Disjoint (range f) (range g)) : ker (f.coprod g) = (ker f).prod (ker g) := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    ⊢ Eq (LinearMap.ker (f.coprod g)) ((LinearMap.ker f).prod (LinearMap.ker g))
  -/
  apply le_antisymm _ (ker_prod_ker_le_ker_coprod f g)
  /-
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    ⊢ LE.le (LinearMap.ker (f.coprod g)) ((LinearMap.ker f).prod (LinearMap.ker g))
  -/
  rintro ⟨y, z⟩ h
  /-
    case mk
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    y : M
    z : M₂
    h : Membership.mem (LinearMap.ker (f.coprod g)) { fst := y, snd := z }
    ⊢ Membership.mem ((LinearMap.ker f).prod (LinearMap.ker g)) { fst := y, snd := …
  -/
  simp only [mem_ker, mem_prod, coprod_apply] at h ⊢
  have : f y ∈ (range f) ⊓ (range g) := by
    simp only [true_and, mem_range, mem_inf, exists_apply_eq_apply]
    use -z
    rwa [eq_comm, map_neg, ← sub_eq_zero, sub_neg_eq_add]
  /-
    case mk
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    y : M
    z : M₂
    h : Eq (HAdd.hAdd (f y) (g z)) 0
    this : Membership.mem (Min.min (LinearMap.range f) (LinearMap.range g)) (f y)
    ⊢ And (Eq (f y) 0) (Eq (g z) 0)
  -/
  rw [hd.eq_bot, mem_bot] at this
  /-
    case mk
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    y : M
    z : M₂
    h : Eq (HAdd.hAdd (f y) (g z)) 0
    this : Eq (f y) 0
    ⊢ And (Eq (f y) 0) (Eq (g z) 0)
  -/
  rw [this] at h
  /-
    case mk
    R : Type u
    M : Type v
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    M₂ : Type u_3
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    M₃ : Type u_4
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₃
    g : LinearMap (RingHom.id R) M₂ M₃
    hd : Disjoint (LinearMap.range f) (LinearMap.range g)
    y : M
    z : M₂
    h : Eq (HAdd.hAdd 0 (g z)) 0
    this : Eq (f y) 0
    ⊢ And (Eq (f y) 0) (Eq (g z) 0)
  -/
  simpa [this] using h
  /-
    🎉 no goals
  -/


theorem sup_eq_range (p q : Submodule R M) : p ⊔ q = range (p.subtype.coprod q.subtype) :=
                            /-
                              R : Type u
                              M : Type v
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              p q : Submodule R M
                              x : M
                              ⊢ Iff (Membership.mem (Max.max p q) x) (Membership.mem (LinearMap.range (p.sub …
                            -/
  Submodule.ext fun x => by simp [Submodule.mem_sup, SetLike.exists]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem map_inl : p.map (inl R M M₂) = prod p ⊥ := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p : Submodule R M
    ⊢ Eq (Submodule.map (LinearMap.inl R M M₂) p) (p.prod Bot.bot)
  -/
  ext ⟨x, y⟩
  simp only [and_left_comm, eq_comm, mem_map, Prod.mk.inj_iff, inl_apply, mem_bot, exists_eq_left',
    mem_prod]


@[simp]
theorem map_inr : q.map (inr R M M₂) = prod ⊥ q := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    q : Submodule R M₂
    ⊢ Eq (Submodule.map (LinearMap.inr R M M₂) q) (Bot.bot.prod q)
  -/
  ext ⟨x, y⟩; simp [and_left_comm, eq_comm, and_comm]
              /-
                🎉 no goals
              -/


@[simp]
                                                          /-
                                                            R : Type u
                                                            M : Type v
                                                            M₂ : Type w
                                                            inst✝⁴ : Semiring R
                                                            inst✝³ : AddCommMonoid M
                                                            inst✝² : AddCommMonoid M₂
                                                            inst✝¹ : Module R M
                                                            inst✝ : Module R M₂
                                                            p : Submodule R M
                                                            ⊢ Eq (Submodule.comap (LinearMap.fst R M M₂) p) (p.prod Top.top)
                                                          -/
theorem comap_fst : p.comap (fst R M M₂) = prod p ⊤ := by ext ⟨x, y⟩; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                          /-
                                                            R : Type u
                                                            M : Type v
                                                            M₂ : Type w
                                                            inst✝⁴ : Semiring R
                                                            inst✝³ : AddCommMonoid M
                                                            inst✝² : AddCommMonoid M₂
                                                            inst✝¹ : Module R M
                                                            inst✝ : Module R M₂
                                                            q : Submodule R M₂
                                                            ⊢ Eq (Submodule.comap (LinearMap.snd R M M₂) q) (Top.top.prod q)
                                                          -/
theorem comap_snd : q.comap (snd R M M₂) = prod ⊤ q := by ext ⟨x, y⟩; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                 /-
                                                                   R : Type u
                                                                   M : Type v
                                                                   M₂ : Type w
                                                                   inst✝⁴ : Semiring R
                                                                   inst✝³ : AddCommMonoid M
                                                                   inst✝² : AddCommMonoid M₂
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : Module R M₂
                                                                   p : Submodule R M
                                                                   q : Submodule R M₂
                                                                   ⊢ Eq (Submodule.comap (LinearMap.inl R M M₂) (p.prod q)) p
                                                                 -/
theorem prod_comap_inl : (prod p q).comap (inl R M M₂) = p := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                 /-
                                                                   R : Type u
                                                                   M : Type v
                                                                   M₂ : Type w
                                                                   inst✝⁴ : Semiring R
                                                                   inst✝³ : AddCommMonoid M
                                                                   inst✝² : AddCommMonoid M₂
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : Module R M₂
                                                                   p : Submodule R M
                                                                   q : Submodule R M₂
                                                                   ⊢ Eq (Submodule.comap (LinearMap.inr R M M₂) (p.prod q)) q
                                                                 -/
theorem prod_comap_inr : (prod p q).comap (inr R M M₂) = q := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem prod_map_fst : (prod p q).map (fst R M M₂) = p := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p : Submodule R M
    q : Submodule R M₂
    ⊢ Eq (Submodule.map (LinearMap.fst R M M₂) (p.prod q)) p
  -/
  ext x; simp [(⟨0, zero_mem _⟩ : ∃ x, x ∈ q)]
         /-
           🎉 no goals
         -/


@[simp]
theorem prod_map_snd : (prod p q).map (snd R M M₂) = q := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p : Submodule R M
    q : Submodule R M₂
    ⊢ Eq (Submodule.map (LinearMap.snd R M M₂) (p.prod q)) q
  -/
  ext x; simp [(⟨0, zero_mem _⟩ : ∃ x, x ∈ p)]
         /-
           🎉 no goals
         -/


@[simp]
                                             /-
                                               R : Type u
                                               M : Type v
                                               M₂ : Type w
                                               inst✝⁴ : Semiring R
                                               inst✝³ : AddCommMonoid M
                                               inst✝² : AddCommMonoid M₂
                                               inst✝¹ : Module R M
                                               inst✝ : Module R M₂
                                               ⊢ Eq (LinearMap.ker (LinearMap.inl R M M₂)) Bot.bot
                                             -/
theorem ker_inl : ker (inl R M M₂) = ⊥ := by rw [ker, ← prod_bot, prod_comap_inl]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                             /-
                                               R : Type u
                                               M : Type v
                                               M₂ : Type w
                                               inst✝⁴ : Semiring R
                                               inst✝³ : AddCommMonoid M
                                               inst✝² : AddCommMonoid M₂
                                               inst✝¹ : Module R M
                                               inst✝ : Module R M₂
                                               ⊢ Eq (LinearMap.ker (LinearMap.inr R M M₂)) Bot.bot
                                             -/
theorem ker_inr : ker (inr R M M₂) = ⊥ := by rw [ker, ← prod_bot, prod_comap_inr]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                                 /-
                                                   R : Type u
                                                   M : Type v
                                                   M₂ : Type w
                                                   inst✝⁴ : Semiring R
                                                   inst✝³ : AddCommMonoid M
                                                   inst✝² : AddCommMonoid M₂
                                                   inst✝¹ : Module R M
                                                   inst✝ : Module R M₂
                                                   ⊢ Eq (LinearMap.range (LinearMap.fst R M M₂)) Top.top
                                                 -/
theorem range_fst : range (fst R M M₂) = ⊤ := by rw [range_eq_map, ← prod_top, prod_map_fst]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                 /-
                                                   R : Type u
                                                   M : Type v
                                                   M₂ : Type w
                                                   inst✝⁴ : Semiring R
                                                   inst✝³ : AddCommMonoid M
                                                   inst✝² : AddCommMonoid M₂
                                                   inst✝¹ : Module R M
                                                   inst✝ : Module R M₂
                                                   ⊢ Eq (LinearMap.range (LinearMap.snd R M M₂)) Top.top
                                                 -/
theorem range_snd : range (snd R M M₂) = ⊤ := by rw [range_eq_map, ← prod_top, prod_map_snd]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- `M` as a submodule of `M × N`. -/
def fst : Submodule R (M × M₂) :=
  (⊥ : Submodule R M₂).comap (LinearMap.snd R M M₂)


/-- `M` as a submodule of `M × N` is isomorphic to `M`. -/
@[simps]
def fstEquiv : Submodule.fst R M M₂ ≃ₗ[R] M where
  -- Porting note: proofs were `tidy` or `simp`
  toFun x := x.1.1
                          /-
                            R : Type u
                            K : Type u'
                            M : Type v
                            V : Type v'
                            M₂ : Type w
                            V₂ : Type w'
                            M₃ : Type y
                            V₃ : Type y'
                            M₄ : Type z
                            ι : Type x
                            M₅ : Type u_1
                            M₆ : Type u_2
                            inst✝⁴ : Semiring R
                            inst✝³ : AddCommMonoid M
                            inst✝² : AddCommMonoid M₂
                            inst✝¹ : Module R M
                            inst✝ : Module R M₂
                            p : Submodule R M
                            q : Submodule R M₂
                            m : M
                            ⊢ Membership.mem (Submodule.fst R M M₂) { fst := m, snd := 0 }
                          -/
                 /-
                   R : Type u
                   K : Type u'
                   M : Type v
                   V : Type v'
                   M₂ : Type w
                   V₂ : Type w'
                   M₃ : Type y
                   V₃ : Type y'
                   M₄ : Type z
                   ι : Type x
                   M₅ : Type u_1
                   M₆ : Type u_2
                   inst✝⁴ : Semiring R
                   inst✝³ : AddCommMonoid M
                   inst✝² : AddCommMonoid M₂
                   inst✝¹ : Module R M
                   inst✝ : Module R M₂
                   p : Submodule R M
                   q : Submodule R M₂
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem (Submodule.fst R M M₂) x), Eq ((fun …
                 -/
  invFun m := ⟨⟨m, 0⟩, by simp only [fst, comap_bot, mem_ker, snd_apply]⟩
                 /-
                   🎉 no goals
                 -/
                          /-
                            🎉 no goals
                          -/
  map_add' := by simp only [coe_add, Prod.fst_add, implies_true]
  map_smul' := by simp only [SetLike.val_smul, Prod.smul_fst, RingHom.id_apply, Subtype.forall,
    implies_true]
  left_inv := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      ⊢ Function.LeftInverse (fun m => ⟨{ fst := m, snd := 0 }, ⋯⟩) { toFun := fun x …
    -/
    rintro ⟨⟨x, y⟩, hy⟩
    /-
      case mk.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x : M
      y : M₂
      hy : Membership.mem (Submodule.fst R M M₂) { fst := x, snd := y }
      ⊢ Eq ((fun m => ⟨{ fst := m, snd := 0 }, ⋯⟩) ({ toFun := fun x => (↑x).1, map_ …
    -/
    simp only [fst, comap_bot, mem_ker, snd_apply] at hy
    /-
      case mk.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x : M
      y : M₂
      hy✝ : Membership.mem (Submodule.fst R M M₂) { fst := x, snd := y }
      hy : Eq y 0
      ⊢ Eq ((fun m => ⟨{ fst := m, snd := 0 }, ⋯⟩) ({ toFun := fun x => (↑x).1, map_ …
    -/
    simpa only [Subtype.mk.injEq, Prod.mk.injEq, true_and] using hy.symm
    /-
      🎉 no goals
    -/
                  /-
                    R : Type u
                    K : Type u'
                    M : Type v
                    V : Type v'
                    M₂ : Type w
                    V₂ : Type w'
                    M₃ : Type y
                    V₃ : Type y'
                    M₄ : Type z
                    ι : Type x
                    M₅ : Type u_1
                    M₆ : Type u_2
                    inst✝⁴ : Semiring R
                    inst✝³ : AddCommMonoid M
                    inst✝² : AddCommMonoid M₂
                    inst✝¹ : Module R M
                    inst✝ : Module R M₂
                    p : Submodule R M
                    q : Submodule R M₂
                    ⊢ Function.RightInverse (fun m => ⟨{ fst := m, snd := 0 }, ⋯⟩) { toFun := fun  …
                  -/
  right_inv := by rintro x; rfl
                            /-
                              🎉 no goals
                            -/


theorem fst_map_fst : (Submodule.fst R M M₂).map (LinearMap.fst R M M₂) = ⊤ := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10936): was `tidy`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Submodule.map (LinearMap.fst R M M₂) (Submodule.fst R M M₂)) Top.top
  -/
  rw [eq_top_iff]; rintro x -
  simp only [fst, comap_bot, mem_map, mem_ker, snd_apply, fst_apply,
    Prod.exists, exists_eq_left, exists_eq]


theorem fst_map_snd : (Submodule.fst R M M₂).map (LinearMap.snd R M M₂) = ⊥ := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10936): was `tidy`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Submodule.map (LinearMap.snd R M M₂) (Submodule.fst R M M₂)) Bot.bot
  -/
  rw [eq_bot_iff]; intro x
  simp only [fst, comap_bot, mem_map, mem_ker, snd_apply, eq_comm, Prod.exists, exists_eq_left,
    exists_const, mem_bot, imp_self]


/-- `N` as a submodule of `M × N`. -/
def snd : Submodule R (M × M₂) :=
  (⊥ : Submodule R M).comap (LinearMap.fst R M M₂)


/-- `N` as a submodule of `M × N` is isomorphic to `N`. -/
@[simps]
def sndEquiv : Submodule.snd R M M₂ ≃ₗ[R] M₂ where
  -- Porting note: proofs were `tidy` or `simp`
  toFun x := x.1.2
                          /-
                            R : Type u
                            K : Type u'
                            M : Type v
                            V : Type v'
                            M₂ : Type w
                            V₂ : Type w'
                            M₃ : Type y
                            V₃ : Type y'
                            M₄ : Type z
                            ι : Type x
                            M₅ : Type u_1
                            M₆ : Type u_2
                            inst✝⁴ : Semiring R
                            inst✝³ : AddCommMonoid M
                            inst✝² : AddCommMonoid M₂
                            inst✝¹ : Module R M
                            inst✝ : Module R M₂
                            p : Submodule R M
                            q : Submodule R M₂
                            n : M₂
                            ⊢ Membership.mem (Submodule.snd R M M₂) { fst := 0, snd := n }
                          -/
                 /-
                   R : Type u
                   K : Type u'
                   M : Type v
                   V : Type v'
                   M₂ : Type w
                   V₂ : Type w'
                   M₃ : Type y
                   V₃ : Type y'
                   M₄ : Type z
                   ι : Type x
                   M₅ : Type u_1
                   M₆ : Type u_2
                   inst✝⁴ : Semiring R
                   inst✝³ : AddCommMonoid M
                   inst✝² : AddCommMonoid M₂
                   inst✝¹ : Module R M
                   inst✝ : Module R M₂
                   p : Submodule R M
                   q : Submodule R M₂
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem (Submodule.snd R M M₂) x), Eq ((fun …
                 -/
  invFun n := ⟨⟨0, n⟩, by simp only [snd, comap_bot, mem_ker, fst_apply]⟩
                 /-
                   🎉 no goals
                 -/
                          /-
                            🎉 no goals
                          -/
  map_add' := by simp only [coe_add, Prod.snd_add, implies_true]
  map_smul' := by simp only [SetLike.val_smul, Prod.smul_snd, RingHom.id_apply, Subtype.forall,
    implies_true]
  left_inv := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      ⊢ Function.LeftInverse (fun n => ⟨{ fst := 0, snd := n }, ⋯⟩) { toFun := fun x …
    -/
    rintro ⟨⟨x, y⟩, hx⟩
    /-
      case mk.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x : M
      y : M₂
      hx : Membership.mem (Submodule.snd R M M₂) { fst := x, snd := y }
      ⊢ Eq ((fun n => ⟨{ fst := 0, snd := n }, ⋯⟩) ({ toFun := fun x => (↑x).2, map_ …
    -/
    simp only [snd, comap_bot, mem_ker, fst_apply] at hx
    /-
      case mk.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x : M
      y : M₂
      hx✝ : Membership.mem (Submodule.snd R M M₂) { fst := x, snd := y }
      hx : Eq x 0
      ⊢ Eq ((fun n => ⟨{ fst := 0, snd := n }, ⋯⟩) ({ toFun := fun x => (↑x).2, map_ …
    -/
    simpa only [Subtype.mk.injEq, Prod.mk.injEq, and_true] using hx.symm
    /-
      🎉 no goals
    -/
                  /-
                    R : Type u
                    K : Type u'
                    M : Type v
                    V : Type v'
                    M₂ : Type w
                    V₂ : Type w'
                    M₃ : Type y
                    V₃ : Type y'
                    M₄ : Type z
                    ι : Type x
                    M₅ : Type u_1
                    M₆ : Type u_2
                    inst✝⁴ : Semiring R
                    inst✝³ : AddCommMonoid M
                    inst✝² : AddCommMonoid M₂
                    inst✝¹ : Module R M
                    inst✝ : Module R M₂
                    p : Submodule R M
                    q : Submodule R M₂
                    ⊢ Function.RightInverse (fun n => ⟨{ fst := 0, snd := n }, ⋯⟩) { toFun := fun  …
                  -/
  right_inv := by rintro x; rfl
                            /-
                              🎉 no goals
                            -/


theorem snd_map_fst : (Submodule.snd R M M₂).map (LinearMap.fst R M M₂) = ⊥ := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10936): was `tidy`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Submodule.map (LinearMap.fst R M M₂) (Submodule.snd R M M₂)) Bot.bot
  -/
  rw [eq_bot_iff]; intro x
  simp only [snd, comap_bot, mem_map, mem_ker, fst_apply, eq_comm, Prod.exists, exists_eq_left,
    exists_const, mem_bot, imp_self]


theorem snd_map_snd : (Submodule.snd R M M₂).map (LinearMap.snd R M M₂) = ⊤ := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10936): was `tidy`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Submodule.map (LinearMap.snd R M M₂) (Submodule.snd R M M₂)) Top.top
  -/
  rw [eq_top_iff]; rintro x -
  simp only [snd, comap_bot, mem_map, mem_ker, snd_apply, fst_apply,
    Prod.exists, exists_eq_right, exists_eq]


theorem fst_sup_snd : Submodule.fst R M M₂ ⊔ Submodule.snd R M M₂ = ⊤ := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂)) Top.top
  -/
  rw [eq_top_iff]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ LE.le Top.top (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂))
  -/
  rintro ⟨m, n⟩ -
  /-
    case mk
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    m : M
    n : M₂
    ⊢ Membership.mem (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂)) { fst …
  -/
  rw [show (m, n) = (m, 0) + (0, n) by simp]
  /-
    case mk
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    m : M
    n : M₂
    ⊢ Membership.mem (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂)) (HAdd …
  -/
  apply Submodule.add_mem (Submodule.fst R M M₂ ⊔ Submodule.snd R M M₂)
    /-
      case mk.h₁
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      m : M
      n : M₂
      ⊢ Membership.mem (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂)) { fst …
    -/
  · exact Submodule.mem_sup_left (Submodule.mem_comap.mpr (by simp))
    /-
      🎉 no goals
    -/
    /-
      case mk.h₂
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      m : M
      n : M₂
      ⊢ Membership.mem (Max.max (Submodule.fst R M M₂) (Submodule.snd R M M₂)) { fst …
    -/
  · exact Submodule.mem_sup_right (Submodule.mem_comap.mpr (by simp))
    /-
      🎉 no goals
    -/


theorem fst_inf_snd : Submodule.fst R M M₂ ⊓ Submodule.snd R M M₂ = ⊥ := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10936): was `tidy`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq (Min.min (Submodule.fst R M M₂) (Submodule.snd R M M₂)) Bot.bot
  -/
  rw [eq_bot_iff]; rintro ⟨x, y⟩
  simp only [fst, comap_bot, snd, mem_inf, mem_ker, snd_apply, fst_apply, mem_bot,
    Prod.mk_eq_zero, and_comm, imp_self]


theorem le_prod_iff {p₁ : Submodule R M} {p₂ : Submodule R M₂} {q : Submodule R (M × M₂)} :
    q ≤ p₁.prod p₂ ↔ map (LinearMap.fst R M M₂) q ≤ p₁ ∧ map (LinearMap.snd R M M₂) q ≤ p₂ := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p₁ : Submodule R M
    p₂ : Submodule R M₂
    q : Submodule R (Prod M M₂)
    ⊢ Iff (LE.le q (p₁.prod p₂)) (And (LE.le (Submodule.map (LinearMap.fst R M M₂) …
  -/
  constructor
    /-
      case mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      ⊢ LE.le q (p₁.prod p₂) → And (LE.le (Submodule.map (LinearMap.fst R M M₂) q) p …
    -/
  · intro h
    /-
      case mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      h : LE.le q (p₁.prod p₂)
      ⊢ And (LE.le (Submodule.map (LinearMap.fst R M M₂) q) p₁) (LE.le (Submodule.ma …
    -/
    constructor
      /-
        case mp.left
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le q (p₁.prod p₂)
        ⊢ LE.le (Submodule.map (LinearMap.fst R M M₂) q) p₁
      -/
    · rintro x ⟨⟨y1, y2⟩, ⟨hy1, rfl⟩⟩
      /-
        case mp.left.intro.mk.intro
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le q (p₁.prod p₂)
        y1 : M
        y2 : M₂
        hy1 : Membership.mem ↑q { fst := y1, snd := y2 }
        ⊢ Membership.mem p₁ ((LinearMap.fst R M M₂) { fst := y1, snd := y2 })
      -/
      exact (h hy1).1
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le q (p₁.prod p₂)
        ⊢ LE.le (Submodule.map (LinearMap.snd R M M₂) q) p₂
      -/
    · rintro x ⟨⟨y1, y2⟩, ⟨hy1, rfl⟩⟩
      /-
        case mp.right.intro.mk.intro
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le q (p₁.prod p₂)
        y1 : M
        y2 : M₂
        hy1 : Membership.mem ↑q { fst := y1, snd := y2 }
        ⊢ Membership.mem p₂ ((LinearMap.snd R M M₂) { fst := y1, snd := y2 })
      -/
      exact (h hy1).2
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      ⊢ And (LE.le (Submodule.map (LinearMap.fst R M M₂) q) p₁) (LE.le (Submodule.ma …
    -/
  · rintro ⟨hH, hK⟩ ⟨x1, x2⟩ h
    /-
      case mpr.intro.mk
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      hH : LE.le (Submodule.map (LinearMap.fst R M M₂) q) p₁
      hK : LE.le (Submodule.map (LinearMap.snd R M M₂) q) p₂
      x1 : M
      x2 : M₂
      h : Membership.mem q { fst := x1, snd := x2 }
      ⊢ Membership.mem (p₁.prod p₂) { fst := x1, snd := x2 }
    -/
    exact ⟨hH ⟨_, h, rfl⟩, hK ⟨_, h, rfl⟩⟩
    /-
      🎉 no goals
    -/


theorem prod_le_iff {p₁ : Submodule R M} {p₂ : Submodule R M₂} {q : Submodule R (M × M₂)} :
    p₁.prod p₂ ≤ q ↔ map (LinearMap.inl R M M₂) p₁ ≤ q ∧ map (LinearMap.inr R M M₂) p₂ ≤ q := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p₁ : Submodule R M
    p₂ : Submodule R M₂
    q : Submodule R (Prod M M₂)
    ⊢ Iff (LE.le (p₁.prod p₂) q) (And (LE.le (Submodule.map (LinearMap.inl R M M₂) …
  -/
  constructor
    /-
      case mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      ⊢ LE.le (p₁.prod p₂) q → And (LE.le (Submodule.map (LinearMap.inl R M M₂) p₁)  …
    -/
  · intro h
    /-
      case mp
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      h : LE.le (p₁.prod p₂) q
      ⊢ And (LE.le (Submodule.map (LinearMap.inl R M M₂) p₁) q) (LE.le (Submodule.ma …
    -/
    constructor
      /-
        case mp.left
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        ⊢ LE.le (Submodule.map (LinearMap.inl R M M₂) p₁) q
      -/
    · rintro _ ⟨x, hx, rfl⟩
      /-
        case mp.left.intro.intro
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        x : M
        hx : Membership.mem (↑p₁) x
        ⊢ Membership.mem q ((LinearMap.inl R M M₂) x)
      -/
      apply h
      /-
        case mp.left.intro.intro.a
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        x : M
        hx : Membership.mem (↑p₁) x
        ⊢ Membership.mem (p₁.prod p₂) ((LinearMap.inl R M M₂) x)
      -/
      exact ⟨hx, zero_mem p₂⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        ⊢ LE.le (Submodule.map (LinearMap.inr R M M₂) p₂) q
      -/
    · rintro _ ⟨x, hx, rfl⟩
      /-
        case mp.right.intro.intro
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        x : M₂
        hx : Membership.mem (↑p₂) x
        ⊢ Membership.mem q ((LinearMap.inr R M M₂) x)
      -/
      apply h
      /-
        case mp.right.intro.intro.a
        R : Type u
        M : Type v
        M₂ : Type w
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid M₂
        inst✝¹ : Module R M
        inst✝ : Module R M₂
        p₁ : Submodule R M
        p₂ : Submodule R M₂
        q : Submodule R (Prod M M₂)
        h : LE.le (p₁.prod p₂) q
        x : M₂
        hx : Membership.mem (↑p₂) x
        ⊢ Membership.mem (p₁.prod p₂) ((LinearMap.inr R M M₂) x)
      -/
      exact ⟨zero_mem p₁, hx⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      ⊢ And (LE.le (Submodule.map (LinearMap.inl R M M₂) p₁) q) (LE.le (Submodule.ma …
    -/
  · rintro ⟨hH, hK⟩ ⟨x1, x2⟩ ⟨h1, h2⟩
    have h1' : (LinearMap.inl R _ _) x1 ∈ q := by
      apply hH
      simpa using h1
    have h2' : (LinearMap.inr R _ _) x2 ∈ q := by
      apply hK
      simpa using h2
    /-
      case mpr.intro.mk.intro
      R : Type u
      M : Type v
      M₂ : Type w
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M
      inst✝ : Module R M₂
      p₁ : Submodule R M
      p₂ : Submodule R M₂
      q : Submodule R (Prod M M₂)
      hH : LE.le (Submodule.map (LinearMap.inl R M M₂) p₁) q
      hK : LE.le (Submodule.map (LinearMap.inr R M M₂) p₂) q
      x1 : M
      x2 : M₂
      h1 : Membership.mem ↑p₁ { fst := x1, snd := x2 }.1
      h2 : Membership.mem ↑p₂ { fst := x1, snd := x2 }.2
      h1' : Membership.mem q ((LinearMap.inl R M M₂) x1)
      h2' : Membership.mem q ((LinearMap.inr R M M₂) x2)
      ⊢ Membership.mem q { fst := x1, snd := x2 }
    -/
    simpa using add_mem h1' h2'
    /-
      🎉 no goals
    -/


theorem prod_eq_bot_iff {p₁ : Submodule R M} {p₂ : Submodule R M₂} :
    p₁.prod p₂ = ⊥ ↔ p₁ = ⊥ ∧ p₂ = ⊥ := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p₁ : Submodule R M
    p₂ : Submodule R M₂
    ⊢ Iff (Eq (p₁.prod p₂) Bot.bot) (And (Eq p₁ Bot.bot) (Eq p₂ Bot.bot))
  -/
  simp only [eq_bot_iff, prod_le_iff, (gc_map_comap _).le_iff_le, comap_bot, ker_inl, ker_inr]
  /-
    🎉 no goals
  -/


theorem prod_eq_top_iff {p₁ : Submodule R M} {p₂ : Submodule R M₂} :
    p₁.prod p₂ = ⊤ ↔ p₁ = ⊤ ∧ p₂ = ⊤ := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    p₁ : Submodule R M
    p₂ : Submodule R M₂
    ⊢ Iff (Eq (p₁.prod p₂) Top.top) (And (Eq p₁ Top.top) (Eq p₂ Top.top))
  -/
  simp only [eq_top_iff, le_prod_iff, ← (gc_map_comap _).le_iff_le, map_top, range_fst, range_snd]
  /-
    🎉 no goals
  -/


/-- Product of modules is commutative up to linear isomorphism. -/
@[simps apply]
def prodComm (R M N : Type*) [Semiring R] [AddCommMonoid M] [AddCommMonoid N] [Module R M]
    [Module R N] : (M × N) ≃ₗ[R] N × M :=
  { AddEquiv.prodComm with
    toFun := Prod.swap
    map_smul' := fun _r ⟨_m, _n⟩ => rfl }


theorem fst_comp_prodComm :
    (LinearMap.fst R M₂ M).comp (prodComm R M M₂).toLinearMap = (LinearMap.snd R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq ((LinearMap.fst R M₂ M).comp ↑(LinearEquiv.prodComm R M M₂)) (LinearMap.s …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem snd_comp_prodComm :
    (LinearMap.snd R M₂ M).comp (prodComm R M M₂).toLinearMap = (LinearMap.fst R M M₂) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    ⊢ Eq ((LinearMap.snd R M₂ M).comp ↑(LinearEquiv.prodComm R M M₂)) (LinearMap.f …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- Product of modules is associative up to linear isomorphism. -/
@[simps apply]
def prodAssoc (R M₁ M₂ M₃ : Type*) [Semiring R]
    [AddCommMonoid M₁] [AddCommMonoid M₂] [AddCommMonoid M₃]
    [Module R M₁] [Module R M₂] [Module R M₃] : ((M₁ × M₂) × M₃) ≃ₗ[R] (M₁ × (M₂ × M₃)) :=
  { AddEquiv.prodAssoc with
    map_smul' := fun _r ⟨_m, _n⟩ => rfl }


theorem fst_comp_prodAssoc :
    (LinearMap.fst R M₁ (M₂ × M₃)).comp (prodAssoc R M₁ M₂ M₃).toLinearMap =
    (LinearMap.fst R M₁ M₂).comp (LinearMap.fst R (M₁ × M₂) M₃) := by
  /-
    R : Type u
    M₂ : Type w
    M₃ : Type y
    M₁ : Type u_3
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    ⊢ Eq ((LinearMap.fst R M₁ (Prod M₂ M₃)).comp ↑(LinearEquiv.prodAssoc R M₁ M₂ M …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem snd_comp_prodAssoc :
    (LinearMap.snd R M₁ (M₂ × M₃)).comp (prodAssoc R M₁ M₂ M₃).toLinearMap =
    (LinearMap.snd R M₁ M₂).prodMap (LinearMap.id : M₃ →ₗ[R] M₃):= by
  /-
    R : Type u
    M₂ : Type w
    M₃ : Type y
    M₁ : Type u_3
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    ⊢ Eq ((LinearMap.snd R M₁ (Prod M₂ M₃)).comp ↑(LinearEquiv.prodAssoc R M₁ M₂ M …
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- Four-way commutativity of `prod`. The name matches `mul_mul_mul_comm`. -/
@[simps apply]
def prodProdProdComm : ((M × M₂) × M₃ × M₄) ≃ₗ[R] (M × M₃) × M₂ × M₄ :=
  { AddEquiv.prodProdProdComm M M₂ M₃ M₄ with
    toFun := fun mnmn => ((mnmn.1.1, mnmn.2.1), (mnmn.1.2, mnmn.2.2))
    invFun := fun mmnn => ((mmnn.1.1, mmnn.2.1), (mmnn.1.2, mmnn.2.2))
    map_smul' := fun _c _mnmn => rfl }


@[simp]
theorem prodProdProdComm_symm :
    (prodProdProdComm R M M₂ M₃ M₄).symm = prodProdProdComm R M M₃ M₂ M₄ :=
  rfl


@[simp]
theorem prodProdProdComm_toAddEquiv :
    (prodProdProdComm R M M₂ M₃ M₄ : _ ≃+ _) = AddEquiv.prodProdProdComm M M₂ M₃ M₄ :=
  rfl


/-- Product of linear equivalences; the maps come from `Equiv.prodCongr`. -/
protected def prod : (M × M₃) ≃ₗ[R] M₂ × M₄ :=
  { e₁.toAddEquiv.prodCongr e₂.toAddEquiv with
    map_smul' := fun c _x => Prod.ext (e₁.map_smulₛₗ c _) (e₂.map_smulₛₗ c _) }


theorem prod_symm : (e₁.prod e₂).symm = e₁.symm.prod e₂.symm :=
  rfl


@[simp]
theorem prod_apply (p) : e₁.prod e₂ p = (e₁ p.1, e₂ p.2) :=
  rfl


@[simp, norm_cast]
theorem coe_prod :
    (e₁.prod e₂ : M × M₃ →ₗ[R] M₂ × M₄) = (e₁ : M →ₗ[R] M₂).prodMap (e₂ : M₃ →ₗ[R] M₄) :=
  rfl


/-- Equivalence given by a block lower diagonal matrix. `e₁` and `e₂` are diagonal square blocks,
  and `f` is a rectangular block below the diagonal. -/
protected def skewProd (f : M →ₗ[R] M₄) : (M × M₃) ≃ₗ[R] M₂ × M₄ :=
  { ((e₁ : M →ₗ[R] M₂).comp (LinearMap.fst R M M₃)).prod
      ((e₂ : M₃ →ₗ[R] M₄).comp (LinearMap.snd R M M₃) +
        f.comp (LinearMap.fst R M M₃)) with
    invFun := fun p : M₂ × M₄ => (e₁.symm p.1, e₂.symm (p.2 - f (e₁.symm p.1)))
                            /-
                              R : Type u
                              K : Type u'
                              M : Type v
                              V : Type v'
                              M₂ : Type w
                              V₂ : Type w'
                              M₃ : Type y
                              V₃ : Type y'
                              M₄ : Type z
                              ι : Type x
                              M₅ : Type u_1
                              M₆ : Type u_2
                              inst✝⁴ : Semiring R
                              inst✝³ : AddCommMonoid M
                              inst✝² : AddCommMonoid M₂
                              inst✝¹ : AddCommMonoid M₃
                              inst✝ : AddCommGroup M₄
                              module_M : Module R M
                              module_M₂ : Module R M₂
                              module_M₃ : Module R M₃
                              module_M₄ : Module R M₄
                              e₁ : LinearEquiv (RingHom.id R) M M₂
                              e₂ : LinearEquiv (RingHom.id R) M₃ M₄
                              f : LinearMap (RingHom.id R) M M₄
                              p : Prod M M₃
                              ⊢ Eq ((fun p => { fst := e₁.symm p.1, snd := e₂.symm (HSub.hSub p.2 (f (e₁.sym …
                            -/
    left_inv := fun p => by simp
                            /-
                              🎉 no goals
                            -/
                             /-
                               R : Type u
                               K : Type u'
                               M : Type v
                               V : Type v'
                               M₂ : Type w
                               V₂ : Type w'
                               M₃ : Type y
                               V₃ : Type y'
                               M₄ : Type z
                               ι : Type x
                               M₅ : Type u_1
                               M₆ : Type u_2
                               inst✝⁴ : Semiring R
                               inst✝³ : AddCommMonoid M
                               inst✝² : AddCommMonoid M₂
                               inst✝¹ : AddCommMonoid M₃
                               inst✝ : AddCommGroup M₄
                               module_M : Module R M
                               module_M₂ : Module R M₂
                               module_M₃ : Module R M₃
                               module_M₄ : Module R M₄
                               e₁ : LinearEquiv (RingHom.id R) M M₂
                               e₂ : LinearEquiv (RingHom.id R) M₃ M₄
                               f : LinearMap (RingHom.id R) M M₄
                               p : Prod M₂ M₄
                               ⊢ Eq (__src✝.toFun ((fun p => { fst := e₁.symm p.1, snd := e₂.symm (HSub.hSub  …
                             -/
    right_inv := fun p => by simp }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem skewProd_apply (f : M →ₗ[R] M₄) (x) : e₁.skewProd e₂ f x = (e₁ x.1, e₂ x.2 + f x.1) :=
  rfl


@[simp]
theorem skewProd_symm_apply (f : M →ₗ[R] M₄) (x) :
    (e₁.skewProd e₂ f).symm x = (e₁.symm x.1, e₂.symm (x.2 - f (e₁.symm x.1))) :=
  rfl


/-- If the union of the kernels `ker f` and `ker g` spans the domain, then the range of
`Prod f g` is equal to the product of `range f` and `range g`. -/
theorem range_prod_eq {f : M →ₗ[R] M₂} {g : M →ₗ[R] M₃} (h : ker f ⊔ ker g = ⊤) :
    range (prod f g) = (range f).prod (range g) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    ⊢ Eq (LinearMap.range (f.prod g)) ((LinearMap.range f).prod (LinearMap.range g))
  -/
  refine le_antisymm (f.range_prod_le g) ?_
  simp only [SetLike.le_def, prod_apply, mem_range, SetLike.mem_coe, mem_prod, exists_imp, and_imp,
    Prod.forall, Pi.prod]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    ⊢ ∀ (a : M₂) (b : M₃) (x : M), Eq (f x) a → ∀ (x : M), Eq (g x) b → Exists fun …
  -/
  rintro _ _ x rfl y rfl
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify `(f := f)`
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    x y : M
    ⊢ Exists fun y_1 => Eq { fst := f y_1, snd := g y_1 } { fst := f x, snd := g y }
  -/
  simp only [Prod.mk.inj_iff, ← sub_mem_ker_iff (f := f)]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    x y : M
    ⊢ Exists fun y_1 => And (Membership.mem (LinearMap.ker f) (HSub.hSub y_1 x)) ( …
  -/
  have : y - x ∈ ker f ⊔ ker g := by simp only [h, mem_top]
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    x y : M
    this : Membership.mem (Max.max (LinearMap.ker f) (LinearMap.ker g)) (HSub.hSub …
    ⊢ Exists fun y_1 => And (Membership.mem (LinearMap.ker f) (HSub.hSub y_1 x)) ( …
  -/
  rcases mem_sup.1 this with ⟨x', hx', y', hy', H⟩
  /-
    case intro.intro.intro.intro
    R : Type u
    M : Type v
    M₂ : Type w
    M₃ : Type y
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M M₃
    h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
    x y : M
    this : Membership.mem (Max.max (LinearMap.ker f) (LinearMap.ker g)) (HSub.hSub …
    x' : M
    hx' : Membership.mem (LinearMap.ker f) x'
    y' : M
    hy' : Membership.mem (LinearMap.ker g) y'
    H : Eq (HAdd.hAdd x' y') (HSub.hSub y x)
    ⊢ Exists fun y_1 => And (Membership.mem (LinearMap.ker f) (HSub.hSub y_1 x)) ( …
  -/
  refine ⟨x' + x, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M M₃
      h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
      x y : M
      this : Membership.mem (Max.max (LinearMap.ker f) (LinearMap.ker g)) (HSub.hSub …
      x' : M
      hx' : Membership.mem (LinearMap.ker f) x'
      y' : M
      hy' : Membership.mem (LinearMap.ker g) y'
      H : Eq (HAdd.hAdd x' y') (HSub.hSub y x)
      ⊢ Membership.mem (LinearMap.ker f) (HSub.hSub (HAdd.hAdd x' x) x)
    -/
  · rwa [add_sub_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      M : Type v
      M₂ : Type w
      M₃ : Type y
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M M₃
      h : Eq (Max.max (LinearMap.ker f) (LinearMap.ker g)) Top.top
      x y : M
      this : Membership.mem (Max.max (LinearMap.ker f) (LinearMap.ker g)) (HSub.hSub …
      x' : M
      hx' : Membership.mem (LinearMap.ker f) x'
      y' : M
      hy' : Membership.mem (LinearMap.ker g) y'
      H : Eq (HAdd.hAdd x' y') (HSub.hSub y x)
      ⊢ Eq (g (HAdd.hAdd x' x)) (g y)
    -/
  · simp [← eq_sub_iff_add_eq.1 H, map_add, add_left_inj, self_eq_add_right, mem_ker.mp hy']
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
/-- An auxiliary construction for `tunnel`.
The composition of `f`, followed by the isomorphism back to `K`,
followed by the inclusion of this submodule back into `M`. -/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tunnelAux (f : M × N →ₗ[R] M) (Kφ : ΣK : Submodule R M, K ≃ₗ[R] M) : M × N →ₗ[R] M :=
  (Kφ.1.subtype.comp Kφ.2.symm.toLinearMap).comp f


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tunnelAux_injective (f : M × N →ₗ[R] M) (i : Injective f)
    (Kφ : ΣK : Submodule R M, K ≃ₗ[R] M) : Injective (tunnelAux f Kφ) :=
  (Subtype.val_injective.comp Kφ.2.symm.injective).comp i


set_option linter.deprecated false in
/-- Auxiliary definition for `tunnel`. -/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tunnel' (f : M × N →ₗ[R] M) (i : Injective f) : ℕ → ΣK : Submodule R M, K ≃ₗ[R] M
  | 0 => ⟨⊤, LinearEquiv.ofTop ⊤ rfl⟩
  | n + 1 =>
    ⟨(Submodule.fst R M N).map (tunnelAux f (tunnel' f i n)),
      ((Submodule.fst R M N).equivMapOfInjective _
        (tunnelAux_injective f i (tunnel' f i n))).symm.trans (Submodule.fstEquiv R M N)⟩


set_option linter.deprecated false in
/-- Give an injective map `f : M × N →ₗ[R] M` we can find a nested sequence of submodules
all isomorphic to `M`.
-/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tunnel (f : M × N →ₗ[R] M) (i : Injective f) : ℕ →o (Submodule R M)ᵒᵈ :=
  -- Note: the hint `(α := _)` had to be added in https://github.com/leanprover-community/mathlib4/pull/8386
  ⟨fun n => OrderDual.toDual (α := Submodule R M) (tunnel' f i n).1,
    monotone_nat_of_le_succ fun n => by
      /-
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        M₅ : Type u_1
        M₆ : Type u_2
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ⊢ LE.le (OrderDual.toDual (f.tunnel' i n).fst) (OrderDual.toDual (f.tunnel' i  …
      -/
      dsimp [tunnel', tunnelAux]
      /-
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        M₅ : Type u_1
        M₆ : Type u_2
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ⊢ LE.le (OrderDual.toDual (f.tunnel' i n).fst) (OrderDual.toDual (Submodule.ma …
      -/
      rw [Submodule.map_comp, Submodule.map_comp]
      /-
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        M₅ : Type u_1
        M₆ : Type u_2
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ⊢ LE.le (OrderDual.toDual (f.tunnel' i n).fst) (OrderDual.toDual (Submodule.ma …
      -/
      apply Submodule.map_subtype_le⟩
      /-
        🎉 no goals
      -/


set_option linter.deprecated false in
/-- Give an injective map `f : M × N →ₗ[R] M` we can find a sequence of submodules
all isomorphic to `N`.
-/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tailing (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) : Submodule R M :=
  (Submodule.snd R M N).map (tunnelAux f (tunnel' f i n))


set_option linter.deprecated false in
/-- Each `tailing f i n` is a copy of `N`. -/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tailingLinearEquiv (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) : tailing f i n ≃ₗ[R] N :=
  ((Submodule.snd R M N).equivMapOfInjective _ (tunnelAux_injective f i (tunnel' f i n))).symm.trans
    (Submodule.sndEquiv R M N)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailing_le_tunnel (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
    tailing f i n ≤ OrderDual.ofDual (α := Submodule R M) (tunnel f i n) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (f.tailing i n) (OrderDual.ofDual ((f.tunnel i) n))
  -/
  dsimp [tailing, tunnelAux]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (Submodule.map (((f.tunnel' i n).fst.subtype.comp ↑(f.tunnel' i n).snd …
  -/
  rw [Submodule.map_comp, Submodule.map_comp]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (Submodule.map (f.tunnel' i n).fst.subtype (Submodule.map (↑(f.tunnel' …
  -/
  apply Submodule.map_subtype_le
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailing_disjoint_tunnel_succ (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
    Disjoint (tailing f i n) (OrderDual.ofDual (α := Submodule R M) <| tunnel f i (n + 1)) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ Disjoint (f.tailing i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
  -/
  rw [disjoint_iff]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ Eq (Min.min (f.tailing i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)) …
  -/
  dsimp [tailing, tunnel, tunnel']
  rw [Submodule.map_inf_eq_map_inf_comap,
    Submodule.comap_map_eq_of_injective (tunnelAux_injective _ i _), inf_comm,
    Submodule.fst_inf_snd, Submodule.map_bot]


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailing_sup_tunnel_succ_le_tunnel (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
    tailing f i n ⊔ (OrderDual.ofDual (α := Submodule R M) <| tunnel f i (n + 1)) ≤
      (OrderDual.ofDual (α := Submodule R M) <| tunnel f i n) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (Max.max (f.tailing i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n  …
  -/
  dsimp [tailing, tunnel, tunnel', tunnelAux]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (Max.max (Submodule.map (((f.tunnel' i n).fst.subtype.comp ↑(f.tunnel' …
  -/
  rw [← Submodule.map_sup, sup_comm, Submodule.fst_sup_snd, Submodule.map_comp, Submodule.map_comp]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ LE.le (Submodule.map (f.tunnel' i n).fst.subtype (Submodule.map (↑(f.tunnel' …
  -/
  apply Submodule.map_subtype_le
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- The supremum of all the copies of `N` found inside the tunnel. -/
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
def tailings (f : M × N →ₗ[R] M) (i : Injective f) : ℕ → Submodule R M :=
  partialSups (tailing f i)


set_option linter.deprecated false in
@[simp, deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailings_zero (f : M × N →ₗ[R] M) (i : Injective f) : tailings f i 0 = tailing f i 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    ⊢ Eq (f.tailings i 0) (f.tailing i 0)
  -/
  simp [tailings]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[simp, deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailings_succ (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
                                                                      /-
                                                                        R : Type u
                                                                        M : Type v
                                                                        inst✝⁴ : Ring R
                                                                        N : Type u_3
                                                                        inst✝³ : AddCommGroup M
                                                                        inst✝² : Module R M
                                                                        inst✝¹ : AddCommGroup N
                                                                        inst✝ : Module R N
                                                                        f : LinearMap (RingHom.id R) (Prod M N) M
                                                                        i : Function.Injective ⇑f
                                                                        n : Nat
                                                                        ⊢ Eq (f.tailings i (HAdd.hAdd n 1)) (Max.max (f.tailings i n) (f.tailing i (HA …
                                                                      -/
    tailings f i (n + 1) = tailings f i n ⊔ tailing f i (n + 1) := by simp [tailings]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailings_disjoint_tunnel (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
    Disjoint (tailings f i n) (OrderDual.ofDual (α := Submodule R M) <| tunnel f i (n + 1)) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) (Prod M N) M
    i : Function.Injective ⇑f
    n : Nat
    ⊢ Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
  -/
  induction' n with n ih
    /-
      case zero
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      N : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) (Prod M N) M
      i : Function.Injective ⇑f
      ⊢ Disjoint (f.tailings i 0) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd 0 1)))
    -/
  · simp only [tailings_zero]
    /-
      case zero
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      N : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) (Prod M N) M
      i : Function.Injective ⇑f
      ⊢ Disjoint (f.tailing i 0) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd 0 1)))
    -/
    apply tailing_disjoint_tunnel_succ
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      N : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) (Prod M N) M
      i : Function.Injective ⇑f
      n : Nat
      ih : Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
      ⊢ Disjoint (f.tailings i (HAdd.hAdd n 1)) (OrderDual.ofDual ((f.tunnel i) (HAd …
    -/
  · simp only [tailings_succ]
    /-
      case succ
      R : Type u
      M : Type v
      inst✝⁴ : Ring R
      N : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) (Prod M N) M
      i : Function.Injective ⇑f
      n : Nat
      ih : Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
      ⊢ Disjoint (Max.max (f.tailings i n) (f.tailing i (HAdd.hAdd n 1))) (OrderDual …
    -/
    refine Disjoint.disjoint_sup_left_of_disjoint_sup_right ?_ ?_
      /-
        case succ.refine_1
        R : Type u
        M : Type v
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ih : Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
        ⊢ Disjoint (f.tailing i (HAdd.hAdd n 1)) (OrderDual.ofDual ((f.tunnel i) (HAdd …
      -/
    · apply tailing_disjoint_tunnel_succ
      /-
        🎉 no goals
      -/
      /-
        case succ.refine_2
        R : Type u
        M : Type v
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ih : Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
        ⊢ Disjoint (f.tailings i n) (Max.max (f.tailing i (HAdd.hAdd n 1)) (OrderDual. …
      -/
    · apply Disjoint.mono_right _ ih
      /-
        R : Type u
        M : Type v
        inst✝⁴ : Ring R
        N : Type u_3
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : LinearMap (RingHom.id R) (Prod M N) M
        i : Function.Injective ⇑f
        n : Nat
        ih : Disjoint (f.tailings i n) (OrderDual.ofDual ((f.tunnel i) (HAdd.hAdd n 1)))
        ⊢ LE.le (Max.max (f.tailing i (HAdd.hAdd n 1)) (OrderDual.ofDual ((f.tunnel i) …
      -/
      apply tailing_sup_tunnel_succ_le_tunnel
      /-
        🎉 no goals
      -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-06-05")]
theorem tailings_disjoint_tailing (f : M × N →ₗ[R] M) (i : Injective f) (n : ℕ) :
    Disjoint (tailings f i n) (tailing f i (n + 1)) :=
  Disjoint.mono_right (tailing_le_tunnel f i _) (tailings_disjoint_tunnel f i _)


/-- Graph of a linear map. -/
def graph : Submodule R (M × M₂) where
  carrier := { p | p.2 = f p.1 }
  add_mem' (ha : _ = _) (hb : _ = _) := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommGroup M₃
      inst✝⁴ : AddCommGroup M₄
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M₄
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M₃ M₄
      a✝ b✝ : Prod M M₂
      ha : Eq a✝.2 (f a✝.1)
      hb : Eq b✝.2 (f b✝.1)
      ⊢ Membership.mem (setOf fun p => Eq p.2 (f p.1)) (HAdd.hAdd a✝ b✝)
    -/
    change _ + _ = f (_ + _)
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommGroup M₃
      inst✝⁴ : AddCommGroup M₄
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M₄
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M₃ M₄
      a✝ b✝ : Prod M M₂
      ha : Eq a✝.2 (f a✝.1)
      hb : Eq b✝.2 (f b✝.1)
      ⊢ Eq (HAdd.hAdd a✝.2 b✝.2) (f (HAdd.hAdd a✝.1 b✝.1))
    -/
    rw [map_add, ha, hb]
    /-
      🎉 no goals
    -/
  zero_mem' := Eq.symm (map_zero f)
  smul_mem' c x (hx : _ = _) := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommGroup M₃
      inst✝⁴ : AddCommGroup M₄
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M₄
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M₃ M₄
      c : R
      x : Prod M M₂
      hx : Eq x.2 (f x.1)
      ⊢ Membership.mem { carrier := setOf fun p => Eq p.2 (f p.1), add_mem' := ⋯, ze …
    -/
    change _ • _ = f (_ • _)
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      M₅ : Type u_1
      M₆ : Type u_2
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommGroup M₃
      inst✝⁴ : AddCommGroup M₄
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M₄
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M₃ M₄
      c : R
      x : Prod M M₂
      hx : Eq x.2 (f x.1)
      ⊢ Eq (HSMul.hSMul c x.2) (f (HSMul.hSMul c x.1))
    -/
    rw [map_smul, hx]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_graph_iff (x : M × M₂) : x ∈ f.graph ↔ x.2 = f x.1 :=
  Iff.rfl


theorem graph_eq_ker_coprod : g.graph = ker ((-g).coprod LinearMap.id) := by
  /-
    R : Type u
    M₃ : Type y
    M₄ : Type z
    inst✝⁴ : Semiring R
    inst✝³ : AddCommGroup M₃
    inst✝² : AddCommGroup M₄
    inst✝¹ : Module R M₃
    inst✝ : Module R M₄
    g : LinearMap (RingHom.id R) M₃ M₄
    ⊢ Eq g.graph (LinearMap.ker ((Neg.neg g).coprod LinearMap.id))
  -/
  ext x
  /-
    case h
    R : Type u
    M₃ : Type y
    M₄ : Type z
    inst✝⁴ : Semiring R
    inst✝³ : AddCommGroup M₃
    inst✝² : AddCommGroup M₄
    inst✝¹ : Module R M₃
    inst✝ : Module R M₄
    g : LinearMap (RingHom.id R) M₃ M₄
    x : Prod M₃ M₄
    ⊢ Iff (Membership.mem g.graph x) (Membership.mem (LinearMap.ker ((Neg.neg g).c …
  -/
  change _ = _ ↔ -g x.1 + x.2 = _
  /-
    case h
    R : Type u
    M₃ : Type y
    M₄ : Type z
    inst✝⁴ : Semiring R
    inst✝³ : AddCommGroup M₃
    inst✝² : AddCommGroup M₄
    inst✝¹ : Module R M₃
    inst✝ : Module R M₄
    g : LinearMap (RingHom.id R) M₃ M₄
    x : Prod M₃ M₄
    ⊢ Iff (Eq x.2 (g x.1)) (Eq (HAdd.hAdd (Neg.neg (g x.1)) x.2) 0)
  -/
  rw [add_comm, add_neg_eq_zero]
  /-
    🎉 no goals
  -/


theorem graph_eq_range_prod : f.graph = range (LinearMap.id.prod f) := by
  /-
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M M₂
    ⊢ Eq f.graph (LinearMap.range (LinearMap.id.prod f))
  -/
  ext x
  /-
    case h
    R : Type u
    M : Type v
    M₂ : Type w
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M M₂
    x : Prod M M₂
    ⊢ Iff (Membership.mem f.graph x) (Membership.mem (LinearMap.range (LinearMap.i …
  -/
  exact ⟨fun hx => ⟨x.1, Prod.ext rfl hx.symm⟩, fun ⟨u, hu⟩ => hu ▸ rfl⟩
  /-
    🎉 no goals
  -/


/-- **Vertical line test** for module homomorphisms.

Let `f : G → H × I` be a linear (or semilinear) map to a product. Assume that `f` is surjective on
the first factor and that the image of `f` intersects every "vertical line" `{(h, i) | i : I}` at
most once. Then the image of `f` is the graph of some linear map `f' : H → I`. -/
lemma LinearMap.exists_range_eq_graph {f : G →ₛₗ[σ] H × I} (hf₁ : Surjective (Prod.fst ∘ f))
    (hf : ∀ g₁ g₂, (f g₁).1 = (f g₂).1 → (f g₁).2 = (f g₂).2) :
    ∃ f' : H →ₗ[S] I, LinearMap.range f = LinearMap.graph f' := by
  obtain ⟨f', hf'⟩ :=
    AddMonoidHom.exists_mrange_eq_mgraph (G := G) (H := H) (I := I) (f := f) hf₁ hf
  simp only [SetLike.ext_iff, AddMonoidHom.mem_mrange, AddMonoidHom.coe_coe,
    AddMonoidHom.mem_mgraph] at hf'
  use
  { toFun := f'.toFun
    map_add' := f'.map_add'
    map_smul' := by
      intro s h
      simp only [ZeroHom.toFun_eq_coe, AddMonoidHom.toZeroHom_coe, RingHom.id_apply]
      refine (hf' (s • h, _)).mp ?_
      rw [← Prod.smul_mk, ← LinearMap.mem_range]
      apply Submodule.smul_mem
      rw [LinearMap.mem_range, hf'] }
  /-
    case h
    R : Type u_3
    S : Type u_4
    G : Type u_5
    H : Type u_6
    I : Type u_7
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring S
    σ : RingHom R S
    inst✝⁶ : RingHomSurjective σ
    inst✝⁵ : AddCommMonoid G
    inst✝⁴ : Module R G
    inst✝³ : AddCommMonoid H
    inst✝² : Module S H
    inst✝¹ : AddCommMonoid I
    inst✝ : Module S I
    f : LinearMap σ G (Prod H I)
    hf₁ : Function.Surjective (Function.comp Prod.fst ⇑f)
    hf : ∀ (g₁ g₂ : G), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    f' : AddMonoidHom H I
    hf' : ∀ (x : Prod H I), Iff (Exists fun x_1 => Eq (f x_1) x) (Eq (f' x.1) x.2)
    ⊢ Eq (LinearMap.range f) { toFun := (↑f').toFun, map_add' := ⋯, map_smul' := ⋯ …
  -/
  ext x
  simpa only [mem_range, Eq.comm, ZeroHom.toFun_eq_coe, AddMonoidHom.toZeroHom_coe, mem_graph_iff,
    coe_mk, AddHom.coe_mk, AddMonoidHom.coe_coe, Set.mem_range] using hf' x


/-- **Vertical line test** for module homomorphisms.

Let `G ≤ H × I` be a submodule of a product of modules. Assume that `G` maps bijectively to the
first factor. Then `G` is the graph of some module homomorphism `f : H →ₗ[R] I`. -/
lemma Submodule.exists_eq_graph {G : Submodule S (H × I)} (hf₁ : Bijective (Prod.fst ∘ G.subtype)) :
    ∃ f : H →ₗ[S] I, G = LinearMap.graph f := by
  simpa only [range_subtype] using LinearMap.exists_range_eq_graph hf₁.surjective
      (fun a b h ↦ congr_arg (Prod.snd ∘ G.subtype) (hf₁.injective h))


/-- **Line test** for module isomorphisms.

Let `f : G → H × I` be a homomorphism to a product of modules. Assume that `f` is surjective onto
both factors and that the image of `f` intersects every "vertical line" `{(h, i) | i : I}` and every
"horizontal line" `{(h, i) | h : H}` at most once. Then the image of `f` is the graph of some
module isomorphism `f' : H ≃ I`. -/
lemma LinearMap.exists_linearEquiv_eq_graph {f : G →ₛₗ[σ] H × I} (hf₁ : Surjective (Prod.fst ∘ f))
    (hf₂ : Surjective (Prod.snd ∘ f)) (hf : ∀ g₁ g₂, (f g₁).1 = (f g₂).1 ↔ (f g₁).2 = (f g₂).2) :
    ∃ e : H ≃ₗ[S] I, range f = e.toLinearMap.graph := by
  /-
    R : Type u_3
    S : Type u_4
    G : Type u_5
    H : Type u_6
    I : Type u_7
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring S
    σ : RingHom R S
    inst✝⁶ : RingHomSurjective σ
    inst✝⁵ : AddCommMonoid G
    inst✝⁴ : Module R G
    inst✝³ : AddCommMonoid H
    inst✝² : Module S H
    inst✝¹ : AddCommMonoid I
    inst✝ : Module S I
    f : LinearMap σ G (Prod H I)
    hf₁ : Function.Surjective (Function.comp Prod.fst ⇑f)
    hf₂ : Function.Surjective (Function.comp Prod.snd ⇑f)
    hf : ∀ (g₁ g₂ : G), Iff (Eq (f g₁).1 (f g₂).1) (Eq (f g₁).2 (f g₂).2)
    ⊢ Exists fun e => Eq (LinearMap.range f) (↑e).graph
  -/
  obtain ⟨e₁, he₁⟩ := f.exists_range_eq_graph hf₁ fun _ _ ↦ (hf _ _).1
  obtain ⟨e₂, he₂⟩ := ((LinearEquiv.prodComm _ _ _).toLinearMap.comp f).exists_range_eq_graph
    (by simpa) <| by simp [hf]
  have he₁₂ h i : e₁ h = i ↔ e₂ i = h := by
    simp only [SetLike.ext_iff, LinearMap.mem_graph_iff] at he₁ he₂
    rw [Eq.comm, ← he₁ (h, i), Eq.comm, ← he₂ (i, h)]
    simp only [mem_range, coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
      LinearEquiv.prodComm_apply, Prod.swap_eq_iff_eq_swap, Prod.swap_prod_mk]
  exact ⟨
  { toFun := e₁
    map_smul' := e₁.map_smul'
    map_add' := e₁.map_add'
    invFun := e₂
    left_inv := fun h ↦ by rw [← he₁₂]
    right_inv := fun i ↦ by rw [he₁₂] }, he₁⟩


/-- **Goursat's lemma** for module isomorphisms.

Let `G ≤ H × I` be a submodule of a product of modules. Assume that the natural maps from `G` to
both factors are bijective. Then `G` is the graph of some module isomorphism `f : H ≃ I`. -/
lemma Submodule.exists_equiv_eq_graph {G : Submodule S (H × I)}
    (hG₁ : Bijective (Prod.fst ∘ G.subtype)) (hG₂ : Bijective (Prod.snd ∘ G.subtype)) :
    ∃ e : H ≃ₗ[S] I, G = e.toLinearMap.graph := by
  simpa only [range_subtype] using LinearMap.exists_linearEquiv_eq_graph
    hG₁.surjective hG₂.surjective fun _ _ ↦ hG₁.injective.eq_iff.trans hG₂.injective.eq_iff.symm


