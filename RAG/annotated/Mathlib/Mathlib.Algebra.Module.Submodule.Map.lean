/-- The pushforward of a submodule `p ⊆ M` by `f : M → M₂` -/
def map (f : F) (p : Submodule R M) : Submodule R₂ M₂ :=
  { p.toAddSubmonoid.map f with
    carrier := f '' p
    smul_mem' := by
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹² : Semiring R
        inst✝¹¹ : Semiring R₂
        inst✝¹⁰ : Semiring R₃
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : AddCommMonoid M₃
        inst✝⁶ : Module R M
        inst✝⁵ : Module R₂ M₂
        inst✝⁴ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        inst✝² : RingHomSurjective σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        p : Submodule R M
        ⊢ ∀ (c : R₂) {x : M₂}, Membership.mem { carrier := Set.image ⇑f ↑p, add_mem' : …
      -/
      rintro c x ⟨y, hy, rfl⟩
      /-
        case intro.intro
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹² : Semiring R
        inst✝¹¹ : Semiring R₂
        inst✝¹⁰ : Semiring R₃
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : AddCommMonoid M₃
        inst✝⁶ : Module R M
        inst✝⁵ : Module R₂ M₂
        inst✝⁴ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        inst✝² : RingHomSurjective σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        p : Submodule R M
        c : R₂
        y : M
        hy : Membership.mem (↑p) y
        ⊢ Membership.mem { carrier := Set.image ⇑f ↑p, add_mem' := ⋯, zero_mem' := ⋯ } …
      -/
      obtain ⟨a, rfl⟩ := σ₁₂.surjective c
      /-
        case intro.intro.intro
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹² : Semiring R
        inst✝¹¹ : Semiring R₂
        inst✝¹⁰ : Semiring R₃
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : AddCommMonoid M₃
        inst✝⁶ : Module R M
        inst✝⁵ : Module R₂ M₂
        inst✝⁴ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        inst✝² : RingHomSurjective σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        p : Submodule R M
        y : M
        hy : Membership.mem (↑p) y
        a : R
        ⊢ Membership.mem { carrier := Set.image ⇑f ↑p, add_mem' := ⋯, zero_mem' := ⋯ } …
      -/
      exact ⟨_, p.smul_mem a hy, map_smulₛₗ f _ _⟩ }
      /-
        🎉 no goals
      -/


@[simp]
theorem map_coe (f : F) (p : Submodule R M) : (map f p : Set M₂) = f '' p :=
  rfl


theorem map_toAddSubmonoid (f : M →ₛₗ[σ₁₂] M₂) (p : Submodule R M) :
    (p.map f).toAddSubmonoid = p.toAddSubmonoid.map (f : M →+ M₂) :=
  SetLike.coe_injective rfl


theorem map_toAddSubmonoid' (f : M →ₛₗ[σ₁₂] M₂) (p : Submodule R M) :
    (p.map f).toAddSubmonoid = p.toAddSubmonoid.map f :=
  SetLike.coe_injective rfl


@[simp]
theorem _root_.AddMonoidHom.coe_toIntLinearMap_map {A A₂ : Type*} [AddCommGroup A] [AddCommGroup A₂]
    (f : A →+ A₂) (s : AddSubgroup A) :
    (AddSubgroup.toIntSubmodule s).map f.toIntLinearMap =
      AddSubgroup.toIntSubmodule (s.map f) := rfl


@[simp]
theorem _root_.MonoidHom.coe_toAdditive_map {G G₂ : Type*} [Group G] [Group G₂] (f : G →* G₂)
    (s : Subgroup G) :
    s.toAddSubgroup.map (MonoidHom.toAdditive f) = Subgroup.toAddSubgroup (s.map f) := rfl


@[simp]
theorem _root_.AddMonoidHom.coe_toMultiplicative_map {G G₂ : Type*} [AddGroup G] [AddGroup G₂]
    (f : G →+ G₂) (s : AddSubgroup G) :
    s.toSubgroup.map (AddMonoidHom.toMultiplicative f) = AddSubgroup.toSubgroup (s.map f) := rfl


@[simp]
theorem mem_map {f : F} {p : Submodule R M} {x : M₂} : x ∈ map f p ↔ ∃ y, y ∈ p ∧ f y = x :=
  Iff.rfl


theorem mem_map_of_mem {f : F} {p : Submodule R M} {r} (h : r ∈ p) : f r ∈ map f p :=
  Set.mem_image_of_mem _ h


theorem apply_coe_mem_map (f : F) {p : Submodule R M} (r : p) : f r ∈ map f p :=
  mem_map_of_mem r.prop


@[simp]
theorem map_id : map (LinearMap.id : M →ₗ[R] M) p = p :=
                            /-
                              R : Type u_1
                              M : Type u_5
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              p : Submodule R M
                              a : M
                              ⊢ Iff (Membership.mem (Submodule.map LinearMap.id p) a) (Membership.mem p a)
                            -/
  Submodule.ext fun a => by simp
                            /-
                              🎉 no goals
                            -/


theorem map_comp [RingHomSurjective σ₂₃] [RingHomSurjective σ₁₃] (f : M →ₛₗ[σ₁₂] M₂)
    (g : M₂ →ₛₗ[σ₂₃] M₃) (p : Submodule R M) : map (g.comp f : M →ₛₗ[σ₁₃] M₃) p = map g (map f p) :=
                              /-
                                R : Type u_1
                                R₂ : Type u_3
                                R₃ : Type u_4
                                M : Type u_5
                                M₂ : Type u_7
                                M₃ : Type u_8
                                inst✝¹² : Semiring R
                                inst✝¹¹ : Semiring R₂
                                inst✝¹⁰ : Semiring R₃
                                inst✝⁹ : AddCommMonoid M
                                inst✝⁸ : AddCommMonoid M₂
                                inst✝⁷ : AddCommMonoid M₃
                                inst✝⁶ : Module R M
                                inst✝⁵ : Module R₂ M₂
                                inst✝⁴ : Module R₃ M₃
                                σ₁₂ : RingHom R R₂
                                σ₂₃ : RingHom R₂ R₃
                                σ₁₃ : RingHom R R₃
                                inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                inst✝² : RingHomSurjective σ₁₂
                                inst✝¹ : RingHomSurjective σ₂₃
                                inst✝ : RingHomSurjective σ₁₃
                                f : LinearMap σ₁₂ M M₂
                                g : LinearMap σ₂₃ M₂ M₃
                                p : Submodule R M
                                ⊢ Eq ↑(Submodule.map (g.comp f) p) ↑(Submodule.map g (Submodule.map f p))
                              -/
  SetLike.coe_injective <| by simp only [← image_comp, map_coe, LinearMap.coe_comp, comp_apply]
                              /-
                                🎉 no goals
                              -/


@[gcongr]
theorem map_mono {f : F} {p p' : Submodule R M} : p ≤ p' → map f p ≤ map f p' :=
  image_subset _


@[simp]
protected theorem map_zero : map (0 : M →ₛₗ[σ₁₂] M₂) p = ⊥ :=
  have : ∃ x : M, x ∈ p := ⟨0, p.zero_mem⟩
            /-
              R : Type u_1
              R₂ : Type u_3
              M : Type u_5
              M₂ : Type u_7
              inst✝⁶ : Semiring R
              inst✝⁵ : Semiring R₂
              inst✝⁴ : AddCommMonoid M
              inst✝³ : AddCommMonoid M₂
              inst✝² : Module R M
              inst✝¹ : Module R₂ M₂
              σ₁₂ : RingHom R R₂
              p : Submodule R M
              inst✝ : RingHomSurjective σ₁₂
              this : Exists fun x => Membership.mem p x
              ⊢ ∀ (x : M₂), Iff (Membership.mem (Submodule.map 0 p) x) (Membership.mem Bot.b …
            -/
  ext <| by simp [this, eq_comm]
            /-
              🎉 no goals
            -/


theorem map_add_le (f g : M →ₛₗ[σ₁₂] M₂) : map (f + g) p ≤ map f p ⊔ map g p := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    p : Submodule R M
    inst✝ : RingHomSurjective σ₁₂
    f g : LinearMap σ₁₂ M M₂
    ⊢ LE.le (Submodule.map (HAdd.hAdd f g) p) (Max.max (Submodule.map f p) (Submod …
  -/
  rintro x ⟨m, hm, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    p : Submodule R M
    inst✝ : RingHomSurjective σ₁₂
    f g : LinearMap σ₁₂ M M₂
    m : M
    hm : Membership.mem (↑p) m
    ⊢ Membership.mem (Max.max (Submodule.map f p) (Submodule.map g p)) ((HAdd.hAdd …
  -/
  exact add_mem_sup (mem_map_of_mem hm) (mem_map_of_mem hm)
  /-
    🎉 no goals
  -/


theorem map_inf_le (f : F) {p q : Submodule R M} :
    (p ⊓ q).map f ≤ p.map f ⊓ q.map f :=
  image_inter_subset f p q


theorem map_inf (f : F) {p q : Submodule R M} (hf : Injective f) :
    (p ⊓ q).map f = p.map f ⊓ q.map f :=
  SetLike.coe_injective <| Set.image_inter hf


lemma map_iInf {ι : Type*} [Nonempty ι] {p : ι → Submodule R M} (f : F) (hf : Injective f) :
    (⨅ i, p i).map f = ⨅ i, (p i).map f :=
                              /-
                                R : Type u_1
                                R₂ : Type u_3
                                M : Type u_5
                                M₂ : Type u_7
                                inst✝⁹ : Semiring R
                                inst✝⁸ : Semiring R₂
                                inst✝⁷ : AddCommMonoid M
                                inst✝⁶ : AddCommMonoid M₂
                                inst✝⁵ : Module R M
                                inst✝⁴ : Module R₂ M₂
                                σ₁₂ : RingHom R R₂
                                inst✝³ : RingHomSurjective σ₁₂
                                F : Type u_9
                                inst✝² : FunLike F M M₂
                                inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
                                ι : Type u_10
                                inst✝ : Nonempty ι
                                p : ι → Submodule R M
                                f : F
                                hf : Function.Injective ⇑f
                                ⊢ Eq ↑(Submodule.map f (iInf fun i => p i)) ↑(iInf fun i => Submodule.map f (p …
                              -/
  SetLike.coe_injective <| by simpa only [map_coe, iInf_coe] using hf.injOn.image_iInter_eq
                              /-
                                🎉 no goals
                              -/


theorem range_map_nonempty (N : Submodule R M) :
    (Set.range (fun ϕ => Submodule.map ϕ N : (M →ₛₗ[σ₁₂] M₂) → Submodule R₂ M₂)).Nonempty :=
  ⟨_, Set.mem_range.mpr ⟨0, rfl⟩⟩


/-- The pushforward of a submodule by an injective linear map is
linearly equivalent to the original submodule. See also `LinearEquiv.submoduleMap` for a
computable version when `f` has an explicit inverse. -/
noncomputable def equivMapOfInjective (f : F) (i : Injective f) (p : Submodule R M) :
    p ≃ₛₗ[σ₁₂] p.map f :=
  { Equiv.Set.image f p i with
    map_add' := by
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        ⊢ ∀ (x y : Subtype fun x => Membership.mem p x), Eq (__src✝.toFun (HAdd.hAdd x …
      -/
      intros
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        x✝ y✝ : Subtype fun x => Membership.mem p x
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (__src✝.toFun x✝) (__src✝.toF …
      -/
      simp only [coe_add, map_add, Equiv.toFun_as_coe, Equiv.Set.image_apply]
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        x✝ y✝ : Subtype fun x => Membership.mem p x
        ⊢ Eq ⟨HAdd.hAdd (f ↑x✝) (f ↑y✝), ⋯⟩ (HAdd.hAdd ⟨f ↑x✝, ⋯⟩ ⟨f ↑y✝, ⋯⟩)
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_smul' := by
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        ⊢ ∀ (m : R) (x : Subtype fun x => Membership.mem p x), Eq ({ toFun := __src✝.t …
      -/
      intros
      -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 changed `map_smulₛₗ` into `map_smulₛₗ _`
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        m✝ : R
        x✝ : Subtype fun x => Membership.mem p x
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul m✝ x✝)) (HSM …
      -/
      simp only [coe_smul_of_tower, map_smulₛₗ _, Equiv.toFun_as_coe, Equiv.Set.image_apply]
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝¹³ : Semiring R
        inst✝¹² : Semiring R₂
        inst✝¹¹ : Semiring R₃
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : AddCommMonoid M₂
        inst✝⁸ : AddCommMonoid M₃
        inst✝⁷ : Module R M
        inst✝⁶ : Module R₂ M₂
        inst✝⁵ : Module R₃ M₃
        σ₁₂ : RingHom R R₂
        σ₂₃ : RingHom R₂ R₃
        σ₁₃ : RingHom R R₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        p✝ p' : Submodule R M
        q q' : Submodule R₂ M₂
        x : M
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        F : Type u_9
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F σ₁₂ M M₂
        f : F
        i : Function.Injective ⇑f
        p : Submodule R M
        m✝ : R
        x✝ : Subtype fun x => Membership.mem p x
        ⊢ Eq ⟨HSMul.hSMul (σ₁₂ m✝) (f ↑x✝), ⋯⟩ (HSMul.hSMul (σ₁₂ m✝) ⟨f ↑x✝, ⋯⟩)
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_equivMapOfInjective_apply (f : F) (i : Injective f) (p : Submodule R M) (x : p) :
    (equivMapOfInjective f i p x : M₂) = f x :=
  rfl


@[simp]
theorem map_equivMapOfInjective_symm_apply (f : F) (i : Injective f) (p : Submodule R M)
    (x : p.map f) : f ((equivMapOfInjective f i p).symm x) = x := by
  rw [← LinearEquiv.apply_symm_apply (equivMapOfInjective f i p) x, coe_equivMapOfInjective_apply,
    i.eq_iff, LinearEquiv.apply_symm_apply]


/-- The pullback of a submodule `p ⊆ M₂` along `f : M → M₂` -/
def comap [SemilinearMapClass F σ₁₂ M M₂] (f : F) (p : Submodule R₂ M₂) : Submodule R M :=
  { p.toAddSubmonoid.comap f with
    carrier := f ⁻¹' p
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 added `map_smulₛₗ _`
                                 /-
                                   R : Type u_1
                                   R₁ : Type u_2
                                   R₂ : Type u_3
                                   R₃ : Type u_4
                                   M : Type u_5
                                   M₁ : Type u_6
                                   M₂ : Type u_7
                                   M₃ : Type u_8
                                   inst✝¹⁴ : Semiring R
                                   inst✝¹³ : Semiring R₂
                                   inst✝¹² : Semiring R₃
                                   inst✝¹¹ : AddCommMonoid M
                                   inst✝¹⁰ : AddCommMonoid M₂
                                   inst✝⁹ : AddCommMonoid M₃
                                   inst✝⁸ : Module R M
                                   inst✝⁷ : Module R₂ M₂
                                   inst✝⁶ : Module R₃ M₃
                                   σ₁₂ : RingHom R R₂
                                   σ₂₃ : RingHom R₂ R₃
                                   σ₁₃ : RingHom R R₃
                                   inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                   p✝ p' : Submodule R M
                                   q q' : Submodule R₂ M₂
                                   x✝ : M
                                   σ₂₁ : RingHom R₂ R
                                   inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
                                   inst✝³ : RingHomInvPair σ₂₁ σ₁₂
                                   F : Type u_9
                                   inst✝² : FunLike F M M₂
                                   inst✝¹ inst✝ : SemilinearMapClass F σ₁₂ M M₂
                                   f : F
                                   p : Submodule R₂ M₂
                                   a : R
                                   x : M
                                   h : Membership.mem { carrier := Set.preimage ⇑f ↑p, add_mem' := ⋯, zero_mem' : …
                                   ⊢ Membership.mem { carrier := Set.preimage ⇑f ↑p, add_mem' := ⋯, zero_mem' :=  …
                                 -/
    smul_mem' := fun a x h => by simp [p.smul_mem (σ₁₂ a) h, map_smulₛₗ _] }
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem comap_coe (f : F) (p : Submodule R₂ M₂) : (comap f p : Set M) = f ⁻¹' p :=
  rfl


@[simp]
theorem AddMonoidHom.coe_toIntLinearMap_comap {A A₂ : Type*} [AddCommGroup A] [AddCommGroup A₂]
    (f : A →+ A₂) (s : AddSubgroup A₂) :
    (AddSubgroup.toIntSubmodule s).comap f.toIntLinearMap =
      AddSubgroup.toIntSubmodule (s.comap f) := rfl


@[simp]
theorem mem_comap {f : F} {p : Submodule R₂ M₂} : x ∈ comap f p ↔ f x ∈ p :=
  Iff.rfl


@[simp]
theorem comap_id : comap (LinearMap.id : M →ₗ[R] M) p = p :=
  SetLike.coe_injective rfl


theorem comap_comp (f : M →ₛₗ[σ₁₂] M₂) (g : M₂ →ₛₗ[σ₂₃] M₃) (p : Submodule R₃ M₃) :
    comap (g.comp f : M →ₛₗ[σ₁₃] M₃) p = comap f (comap g p) :=
  rfl


@[gcongr]
theorem comap_mono {f : F} {q q' : Submodule R₂ M₂} : q ≤ q' → comap f q ≤ comap f q' :=
  preimage_mono


theorem le_comap_pow_of_le_comap (p : Submodule R M) {f : M →ₗ[R] M} (h : p ≤ p.comap f) (k : ℕ) :
    p ≤ p.comap (f ^ k) := by
  induction k with
  | zero => simp [LinearMap.one_eq_id]
  | succ k ih => simp [LinearMap.iterate_succ, comap_comp, h.trans (comap_mono ih)]


theorem map_le_iff_le_comap {f : F} {p : Submodule R M} {q : Submodule R₂ M₂} :
    map f p ≤ q ↔ p ≤ comap f q :=
  image_subset_iff


theorem gc_map_comap (f : F) : GaloisConnection (map f) (comap f)
  | _, _ => map_le_iff_le_comap


@[simp]
theorem map_bot (f : F) : map f ⊥ = ⊥ :=
  (gc_map_comap f).l_bot


@[simp]
theorem map_sup (f : F) : map f (p ⊔ p') = map f p ⊔ map f p' :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_sup


@[simp]
theorem map_iSup {ι : Sort*} (f : F) (p : ι → Submodule R M) :
    map f (⨆ i, p i) = ⨆ i, map f (p i) :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_iSup


@[simp]
theorem comap_top (f : F) : comap f ⊤ = ⊤ :=
  rfl


@[simp]
theorem comap_inf (f : F) : comap f (q ⊓ q') = comap f q ⊓ comap f q' :=
  rfl


@[simp]
theorem comap_iInf [RingHomSurjective σ₁₂] {ι : Sort*} (f : F) (p : ι → Submodule R₂ M₂) :
    comap f (⨅ i, p i) = ⨅ i, comap f (p i) :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).u_iInf


@[simp]
theorem comap_zero : comap (0 : M →ₛₗ[σ₁₂] M₂) q = ⊤ :=
            /-
              R : Type u_1
              R₂ : Type u_3
              M : Type u_5
              M₂ : Type u_7
              inst✝⁵ : Semiring R
              inst✝⁴ : Semiring R₂
              inst✝³ : AddCommMonoid M
              inst✝² : AddCommMonoid M₂
              inst✝¹ : Module R M
              inst✝ : Module R₂ M₂
              σ₁₂ : RingHom R R₂
              q : Submodule R₂ M₂
              ⊢ ∀ (x : M), Iff (Membership.mem (Submodule.comap 0 q) x) (Membership.mem Top. …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


theorem map_comap_le [RingHomSurjective σ₁₂] (f : F) (q : Submodule R₂ M₂) :
    map f (comap f q) ≤ q :=
  (gc_map_comap f).l_u_le _


theorem le_comap_map [RingHomSurjective σ₁₂] (f : F) (p : Submodule R M) : p ≤ comap f (map f p) :=
  (gc_map_comap f).le_u_l _


/-- `map f` and `comap f` form a `GaloisInsertion` when `f` is surjective. -/
def giMapComap (hf : Surjective f) : GaloisInsertion (map f) (comap f) :=
  (gc_map_comap f).toGaloisInsertion fun S x hx => by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x✝ : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Surjective ⇑f
      S : Submodule R₂ M₂
      x : M₂
      hx : Membership.mem S x
      ⊢ Membership.mem (Submodule.map f (Submodule.comap f S)) x
    -/
    rcases hf x with ⟨y, rfl⟩
    /-
      case intro
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Surjective ⇑f
      S : Submodule R₂ M₂
      y : M
      hx : Membership.mem S (f y)
      ⊢ Membership.mem (Submodule.map f (Submodule.comap f S)) (f y)
    -/
    simp only [mem_map, mem_comap]
    /-
      case intro
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Surjective ⇑f
      S : Submodule R₂ M₂
      y : M
      hx : Membership.mem S (f y)
      ⊢ Exists fun y_1 => And (Membership.mem S (f y_1)) (Eq (f y_1) (f y))
    -/
    exact ⟨y, hx, rfl⟩
    /-
      🎉 no goals
    -/


theorem map_comap_eq_of_surjective (p : Submodule R₂ M₂) : (p.comap f).map f = p :=
  (giMapComap hf).l_u_eq _


theorem map_surjective_of_surjective : Function.Surjective (map f) :=
  (giMapComap hf).l_surjective


theorem comap_injective_of_surjective : Function.Injective (comap f) :=
  (giMapComap hf).u_injective


theorem map_sup_comap_of_surjective (p q : Submodule R₂ M₂) :
    (p.comap f ⊔ q.comap f).map f = p ⊔ q :=
  (giMapComap hf).l_sup_u _ _


theorem map_iSup_comap_of_sujective {ι : Sort*} (S : ι → Submodule R₂ M₂) :
    (⨆ i, (S i).comap f).map f = iSup S :=
  (giMapComap hf).l_iSup_u _


theorem map_inf_comap_of_surjective (p q : Submodule R₂ M₂) :
    (p.comap f ⊓ q.comap f).map f = p ⊓ q :=
  (giMapComap hf).l_inf_u _ _


theorem map_iInf_comap_of_surjective {ι : Sort*} (S : ι → Submodule R₂ M₂) :
    (⨅ i, (S i).comap f).map f = iInf S :=
  (giMapComap hf).l_iInf_u _


theorem comap_le_comap_iff_of_surjective {p q : Submodule R₂ M₂} : p.comap f ≤ q.comap f ↔ p ≤ q :=
  (giMapComap hf).u_le_u_iff


lemma comap_lt_comap_iff_of_surjective {p q : Submodule R₂ M₂} : p.comap f < q.comap f ↔ p < q := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    F : Type u_9
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
    inst✝ : RingHomSurjective σ₁₂
    f : F
    hf : Function.Surjective ⇑f
    p q : Submodule R₂ M₂
    ⊢ Iff (LT.lt (Submodule.comap f p) (Submodule.comap f q)) (LT.lt p q)
  -/
                                    /-
                                      🎉 no goals
                                    -/
  apply lt_iff_lt_of_le_iff_le' <;> exact comap_le_comap_iff_of_surjective hf
                                    /-
                                      🎉 no goals
                                    -/


theorem comap_strictMono_of_surjective : StrictMono (comap f) :=
  (giMapComap hf).strictMono_u


/-- `map f` and `comap f` form a `GaloisCoinsertion` when `f` is injective. -/
def gciMapComap (hf : Injective f) : GaloisCoinsertion (map f) (comap f) :=
  (gc_map_comap f).toGaloisCoinsertion fun S x => by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x✝ : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Injective ⇑f
      S : Submodule R M
      x : M
      ⊢ Membership.mem (Submodule.comap f (Submodule.map f S)) x → Membership.mem S x
    -/
    simp only [mem_comap, mem_map, forall_exists_index, and_imp]
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x✝ : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Injective ⇑f
      S : Submodule R M
      x : M
      ⊢ ∀ (x_1 : M), Membership.mem S x_1 → Eq (f x_1) (f x) → Membership.mem S x
    -/
    intro y hy hxy
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x✝ : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Injective ⇑f
      S : Submodule R M
      x y : M
      hy : Membership.mem S y
      hxy : Eq (f y) (f x)
      ⊢ Membership.mem S x
    -/
    rw [hf.eq_iff] at hxy
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      inst✝¹⁴ : Semiring R
      inst✝¹³ : Semiring R₂
      inst✝¹² : Semiring R₃
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : AddCommMonoid M₂
      inst✝⁹ : AddCommMonoid M₃
      inst✝⁸ : Module R M
      inst✝⁷ : Module R₂ M₂
      inst✝⁶ : Module R₃ M₃
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      inst✝⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      p p' : Submodule R M
      q q' : Submodule R₂ M₂
      x✝ : M
      σ₂₁ : RingHom R₂ R
      inst✝⁴ : RingHomInvPair σ₁₂ σ₂₁
      inst✝³ : RingHomInvPair σ₂₁ σ₁₂
      F : Type u_9
      inst✝² : FunLike F M M₂
      inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
      inst✝ : RingHomSurjective σ₁₂
      f : F
      hf : Function.Injective ⇑f
      S : Submodule R M
      x y : M
      hy : Membership.mem S y
      hxy : Eq y x
      ⊢ Membership.mem S x
    -/
    rwa [← hxy]
    /-
      🎉 no goals
    -/


theorem comap_map_eq_of_injective (p : Submodule R M) : (p.map f).comap f = p :=
  (gciMapComap hf).u_l_eq _


theorem comap_surjective_of_injective : Function.Surjective (comap f) :=
  (gciMapComap hf).u_surjective


theorem map_injective_of_injective : Function.Injective (map f) :=
  (gciMapComap hf).l_injective


theorem comap_inf_map_of_injective (p q : Submodule R M) : (p.map f ⊓ q.map f).comap f = p ⊓ q :=
  (gciMapComap hf).u_inf_l _ _


theorem comap_iInf_map_of_injective {ι : Sort*} (S : ι → Submodule R M) :
    (⨅ i, (S i).map f).comap f = iInf S :=
  (gciMapComap hf).u_iInf_l _


theorem comap_sup_map_of_injective (p q : Submodule R M) : (p.map f ⊔ q.map f).comap f = p ⊔ q :=
  (gciMapComap hf).u_sup_l _ _


theorem comap_iSup_map_of_injective {ι : Sort*} (S : ι → Submodule R M) :
    (⨆ i, (S i).map f).comap f = iSup S :=
  (gciMapComap hf).u_iSup_l _


theorem map_le_map_iff_of_injective (p q : Submodule R M) : p.map f ≤ q.map f ↔ p ≤ q :=
  (gciMapComap hf).l_le_l_iff


theorem map_strictMono_of_injective : StrictMono (map f) :=
  (gciMapComap hf).strictMono_l


/-- A linear isomorphism induces an order isomorphism of submodules. -/
@[simps symm_apply apply]
def orderIsoMapComapOfBijective [FunLike F M M₂] [SemilinearMapClass F σ₁₂ M M₂]
    (f : F) (hf : Bijective f) : Submodule R M ≃o Submodule R₂ M₂ where
  toFun := map f
  invFun := comap f
  left_inv := comap_map_eq_of_injective hf.injective
  right_inv := map_comap_eq_of_surjective hf.surjective
  map_rel_iff' := map_le_map_iff_of_injective hf.injective _ _


/-- A linear isomorphism induces an order isomorphism of submodules. -/
@[simps! apply]
def orderIsoMapComap [EquivLike F M M₂] [SemilinearMapClass F σ₁₂ M M₂] (f : F) :
    Submodule R M ≃o Submodule R₂ M₂ := orderIsoMapComapOfBijective f (EquivLike.bijective f)


@[simp]
lemma orderIsoMapComap_symm_apply [EquivLike F M M₂] [SemilinearMapClass F σ₁₂ M M₂]
    (f : F) (p : Submodule R₂ M₂) :
    (orderIsoMapComap f).symm p = comap f p :=
  rfl


theorem map_inf_eq_map_inf_comap [RingHomSurjective σ₁₂] {f : F} {p : Submodule R M}
    {p' : Submodule R₂ M₂} : map f p ⊓ p' = map f (p ⊓ comap f p') :=
                  /-
                    R : Type u_1
                    R₂ : Type u_3
                    M : Type u_5
                    M₂ : Type u_7
                    inst✝⁸ : Semiring R
                    inst✝⁷ : Semiring R₂
                    inst✝⁶ : AddCommMonoid M
                    inst✝⁵ : AddCommMonoid M₂
                    inst✝⁴ : Module R M
                    inst✝³ : Module R₂ M₂
                    σ₁₂ : RingHom R R₂
                    F : Type u_9
                    inst✝² : FunLike F M M₂
                    inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
                    inst✝ : RingHomSurjective σ₁₂
                    f : F
                    p : Submodule R M
                    p' : Submodule R₂ M₂
                    ⊢ LE.le (Min.min (Submodule.map f p) p') (Submodule.map f (Min.min p (Submodul …
                  -/
  le_antisymm (by rintro _ ⟨⟨x, h₁, rfl⟩, h₂⟩; exact ⟨_, ⟨h₁, h₂⟩, rfl⟩)
                                               /-
                                                 🎉 no goals
                                               -/
    (le_inf (map_mono inf_le_left) (map_le_iff_le_comap.2 inf_le_right))


@[simp]
theorem map_comap_subtype : map p.subtype (comap p.subtype p') = p ⊓ p' :=
                   /-
                     R : Type u_1
                     M : Type u_5
                     inst✝² : Semiring R
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     p p' : Submodule R M
                     x : M
                     ⊢ Membership.mem (Submodule.map p.subtype (Submodule.comap p.subtype p')) x →  …
                   -/
  ext fun x => ⟨by rintro ⟨⟨_, h₁⟩, h₂, rfl⟩; exact ⟨h₁, h₂⟩, fun ⟨h₁, h₂⟩ => ⟨⟨_, h₁⟩, h₂, rfl⟩⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem eq_zero_of_bot_submodule : ∀ b : (⊥ : Submodule R M), b = 0
  | ⟨b', hb⟩ => Subtype.eq <| show b' = 0 from (mem_bot R).1 hb


/-- The infimum of a family of invariant submodule of an endomorphism is also an invariant
submodule. -/
theorem _root_.LinearMap.iInf_invariant {σ : R →+* R} [RingHomSurjective σ] {ι : Sort*}
    (f : M →ₛₗ[σ] M) {p : ι → Submodule R M} (hf : ∀ i, ∀ v ∈ p i, f v ∈ p i) :
    ∀ v ∈ iInf p, f v ∈ iInf p := by
  have : ∀ i, (p i).map f ≤ p i := by
    rintro i - ⟨v, hv, rfl⟩
    exact hf i v hv
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    σ : RingHom R R
    inst✝ : RingHomSurjective σ
    ι : Sort u_10
    f : LinearMap σ M M
    p : ι → Submodule R M
    hf : ∀ (i : ι) (v : M), Membership.mem (p i) v → Membership.mem (p i) (f v)
    this : ∀ (i : ι), LE.le (Submodule.map f (p i)) (p i)
    ⊢ ∀ (v : M), Membership.mem (iInf p) v → Membership.mem (iInf p) (f v)
  -/
  suffices (iInf p).map f ≤ iInf p by exact fun v hv => this ⟨v, hv, rfl⟩
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    σ : RingHom R R
    inst✝ : RingHomSurjective σ
    ι : Sort u_10
    f : LinearMap σ M M
    p : ι → Submodule R M
    hf : ∀ (i : ι) (v : M), Membership.mem (p i) v → Membership.mem (p i) (f v)
    this : ∀ (i : ι), LE.le (Submodule.map f (p i)) (p i)
    ⊢ LE.le (Submodule.map f (iInf p)) (iInf p)
  -/
  exact le_iInf fun i => (Submodule.map_mono (iInf_le p i)).trans (this i)
  /-
    🎉 no goals
  -/


theorem disjoint_iff_comap_eq_bot {p q : Submodule R M} : Disjoint p q ↔ comap p.subtype q = ⊥ := by
  rw [← (map_injective_of_injective (show Injective p.subtype from Subtype.coe_injective)).eq_iff,
    map_comap_subtype, map_bot, disjoint_iff]


@[simp]
protected theorem map_neg (f : M →ₗ[R] M₂) : map (-f) p = map f p :=
  ext fun _ =>
    ⟨fun ⟨x, hx, hy⟩ => hy ▸ ⟨-x, show -x ∈ p from neg_mem hx, map_neg f x⟩, fun ⟨x, hx, hy⟩ =>
      hy ▸ ⟨-x, show -x ∈ p from neg_mem hx, (map_neg (-f) _).trans (neg_neg (f x))⟩⟩


@[simp]
lemma comap_neg {f : M →ₗ[R] M₂} {p : Submodule R M₂} :
    p.comap (-f) = p.comap f := by
  /-
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M M₂
    p : Submodule R M₂
    ⊢ Eq (Submodule.comap (Neg.neg f) p) (Submodule.comap f p)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


lemma map_toAddSubgroup (f : M →ₗ[R] M₂) (p : Submodule R M) :
    (p.map f).toAddSubgroup = p.toAddSubgroup.map (f : M →+ M₂) :=
  rfl


theorem comap_smul (f : V →ₗ[K] V₂) (p : Submodule K V₂) (a : K) (h : a ≠ 0) :
    p.comap (a • f) = p.comap f := by
  /-
    K : Type u_9
    V : Type u_10
    V₂ : Type u_11
    inst✝⁴ : Semifield K
    inst✝³ : AddCommMonoid V
    inst✝² : Module K V
    inst✝¹ : AddCommMonoid V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V V₂
    p : Submodule K V₂
    a : K
    h : Ne a 0
    ⊢ Eq (Submodule.comap (HSMul.hSMul a f) p) (Submodule.comap f p)
  -/
  ext b; simp only [Submodule.mem_comap, p.smul_mem_iff h, LinearMap.smul_apply]
         /-
           🎉 no goals
         -/


protected theorem map_smul (f : V →ₗ[K] V₂) (p : Submodule K V) (a : K) (h : a ≠ 0) :
    p.map (a • f) = p.map f :=
                  /-
                    K : Type u_9
                    V : Type u_10
                    V₂ : Type u_11
                    inst✝⁴ : Semifield K
                    inst✝³ : AddCommMonoid V
                    inst✝² : Module K V
                    inst✝¹ : AddCommMonoid V₂
                    inst✝ : Module K V₂
                    f : LinearMap (RingHom.id K) V V₂
                    p : Submodule K V
                    a : K
                    h : Ne a 0
                    ⊢ LE.le (Submodule.map (HSMul.hSMul a f) p) (Submodule.map f p)
                  -/
  le_antisymm (by rw [map_le_iff_le_comap, comap_smul f _ a h, ← map_le_iff_le_comap])
                  /-
                    🎉 no goals
                  -/
        /-
          K : Type u_9
          V : Type u_10
          V₂ : Type u_11
          inst✝⁴ : Semifield K
          inst✝³ : AddCommMonoid V
          inst✝² : Module K V
          inst✝¹ : AddCommMonoid V₂
          inst✝ : Module K V₂
          f : LinearMap (RingHom.id K) V V₂
          p : Submodule K V
          a : K
          h : Ne a 0
          ⊢ LE.le (Submodule.map f p) (Submodule.map (HSMul.hSMul a f) p)
        -/
    (by rw [map_le_iff_le_comap, ← comap_smul f _ a h, ← map_le_iff_le_comap])
        /-
          🎉 no goals
        -/


theorem comap_smul' (f : V →ₗ[K] V₂) (p : Submodule K V₂) (a : K) :
    p.comap (a • f) = ⨅ _ : a ≠ 0, p.comap f := by
  /-
    K : Type u_9
    V : Type u_10
    V₂ : Type u_11
    inst✝⁴ : Semifield K
    inst✝³ : AddCommMonoid V
    inst✝² : Module K V
    inst✝¹ : AddCommMonoid V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V V₂
    p : Submodule K V₂
    a : K
    ⊢ Eq (Submodule.comap (HSMul.hSMul a f) p) (iInf fun x => Submodule.comap f p)
  -/
  classical by_cases h : a = 0 <;> simp [h, comap_smul]
  /-
    🎉 no goals
  -/


theorem map_smul' (f : V →ₗ[K] V₂) (p : Submodule K V) (a : K) :
    p.map (a • f) = ⨆ _ : a ≠ 0, map f p := by
  /-
    K : Type u_9
    V : Type u_10
    V₂ : Type u_11
    inst✝⁴ : Semifield K
    inst✝³ : AddCommMonoid V
    inst✝² : Module K V
    inst✝¹ : AddCommMonoid V₂
    inst✝ : Module K V₂
    f : LinearMap (RingHom.id K) V V₂
    p : Submodule K V
    a : K
    ⊢ Eq (Submodule.map (HSMul.hSMul a f) p) (iSup fun x => Submodule.map f p)
  -/
  classical by_cases h : a = 0 <;> simp [h, Submodule.map_smul]
  /-
    🎉 no goals
  -/


/-- If `s ≤ t`, then we can view `s` as a submodule of `t` by taking the comap
of `t.subtype`. -/
@[simps symm_apply]
def comapSubtypeEquivOfLe {p q : Submodule R M} (hpq : p ≤ q) : comap q.subtype p ≃ₗ[R] p where
  toFun x := ⟨x, x.2⟩
  invFun x := ⟨⟨x, hpq x.2⟩, x.2⟩
                   /-
                     R : Type u_1
                     R₁ : Type u_2
                     R₂ : Type u_3
                     R₃ : Type u_4
                     M : Type u_5
                     M₁ : Type u_6
                     M₂ : Type u_7
                     M₃ : Type u_8
                     inst✝² : Semiring R
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     p q : Submodule R M
                     hpq : LE.le p q
                     x : Subtype fun x => Membership.mem (Submodule.comap q.subtype p) x
                     ⊢ Eq ((fun x => ⟨⟨↑x, ⋯⟩, ⋯⟩) ({ toFun := fun x => ⟨↑↑x, ⋯⟩, map_add' := ⋯, ma …
                   -/
  left_inv x := by simp only [coe_mk, SetLike.eta, LinearEquiv.coe_coe]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u_1
                      R₁ : Type u_2
                      R₂ : Type u_3
                      R₃ : Type u_4
                      M : Type u_5
                      M₁ : Type u_6
                      M₂ : Type u_7
                      M₃ : Type u_8
                      inst✝² : Semiring R
                      inst✝¹ : AddCommMonoid M
                      inst✝ : Module R M
                      p q : Submodule R M
                      hpq : LE.le p q
                      x : Subtype fun x => Membership.mem p x
                      ⊢ Eq ({ toFun := fun x => ⟨↑↑x, ⋯⟩, map_add' := ⋯, map_smul' := ⋯ }.toFun ((fu …
                    -/
  right_inv x := by simp only [Subtype.coe_mk, SetLike.eta, LinearEquiv.coe_coe]
                    /-
                      🎉 no goals
                    -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

-- Porting note: The original theorem generated by `simps` was using `LinearEquiv.toLinearMap`,
-- different from the theorem on Lean 3, and not simp-normal form.

@[simp]
theorem comapSubtypeEquivOfLe_apply_coe {p q : Submodule R M} (hpq : p ≤ q)
    (x : comap q.subtype p) :
    (comapSubtypeEquivOfLe hpq x : M) = (x : M) :=
  rfl


@[simp high]
theorem mem_map_equiv {e : M ≃ₛₗ[τ₁₂] M₂} {x : M₂} :
    x ∈ p.map (e : M →ₛₗ[τ₁₂] M₂) ↔ e.symm x ∈ p := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    τ₂₁ : RingHom R₂ R
    inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
    inst✝ : RingHomInvPair τ₂₁ τ₁₂
    p : Submodule R M
    e : LinearEquiv τ₁₂ M M₂
    x : M₂
    ⊢ Iff (Membership.mem (Submodule.map (↑e) p) x) (Membership.mem p (e.symm x))
  -/
  rw [Submodule.mem_map]; constructor
    /-
      case mp
      R : Type u_1
      R₂ : Type u_3
      M : Type u_5
      M₂ : Type u_7
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring R₂
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      e : LinearEquiv τ₁₂ M M₂
      x : M₂
      ⊢ (Exists fun y => And (Membership.mem p y) (Eq (↑e y) x)) → Membership.mem p  …
    -/
  · rintro ⟨y, hy, hx⟩
    /-
      case mp.intro.intro
      R : Type u_1
      R₂ : Type u_3
      M : Type u_5
      M₂ : Type u_7
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring R₂
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      e : LinearEquiv τ₁₂ M M₂
      x : M₂
      y : M
      hy : Membership.mem p y
      hx : Eq (↑e y) x
      ⊢ Membership.mem p (e.symm x)
    -/
    simp [← hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      R₂ : Type u_3
      M : Type u_5
      M₂ : Type u_7
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring R₂
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      e : LinearEquiv τ₁₂ M M₂
      x : M₂
      ⊢ Membership.mem p (e.symm x) → Exists fun y => And (Membership.mem p y) (Eq ( …
    -/
  · intro hx
    /-
      case mpr
      R : Type u_1
      R₂ : Type u_3
      M : Type u_5
      M₂ : Type u_7
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring R₂
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M
      inst✝² : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      e : LinearEquiv τ₁₂ M M₂
      x : M₂
      hx : Membership.mem p (e.symm x)
      ⊢ Exists fun y => And (Membership.mem p y) (Eq (↑e y) x)
    -/
    exact ⟨e.symm x, hx, by simp⟩
    /-
      🎉 no goals
    -/


theorem map_equiv_eq_comap_symm (e : M ≃ₛₗ[τ₁₂] M₂) (K : Submodule R M) :
    K.map (e : M →ₛₗ[τ₁₂] M₂) = K.comap (e.symm : M₂ →ₛₗ[τ₂₁] M) :=
                            /-
                              R : Type u_1
                              R₂ : Type u_3
                              M : Type u_5
                              M₂ : Type u_7
                              inst✝⁷ : Semiring R
                              inst✝⁶ : Semiring R₂
                              inst✝⁵ : AddCommMonoid M
                              inst✝⁴ : AddCommMonoid M₂
                              inst✝³ : Module R M
                              inst✝² : Module R₂ M₂
                              τ₁₂ : RingHom R R₂
                              τ₂₁ : RingHom R₂ R
                              inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
                              inst✝ : RingHomInvPair τ₂₁ τ₁₂
                              e : LinearEquiv τ₁₂ M M₂
                              K : Submodule R M
                              x✝ : M₂
                              ⊢ Iff (Membership.mem (Submodule.map (↑e) K) x✝) (Membership.mem (Submodule.co …
                            -/
  Submodule.ext fun _ => by rw [mem_map_equiv, mem_comap, LinearEquiv.coe_coe]
                            /-
                              🎉 no goals
                            -/


theorem comap_equiv_eq_map_symm (e : M ≃ₛₗ[τ₁₂] M₂) (K : Submodule R₂ M₂) :
    K.comap (e : M →ₛₗ[τ₁₂] M₂) = K.map (e.symm : M₂ →ₛₗ[τ₂₁] M) :=
  (map_equiv_eq_comap_symm e.symm K).symm


theorem map_symm_eq_iff (e : M ≃ₛₗ[τ₁₂] M₂) {K : Submodule R₂ M₂} :
    K.map e.symm = p ↔ p.map e = K := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    τ₂₁ : RingHom R₂ R
    inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
    inst✝ : RingHomInvPair τ₂₁ τ₁₂
    p : Submodule R M
    e : LinearEquiv τ₁₂ M M₂
    K : Submodule R₂ M₂
    ⊢ Iff (Eq (Submodule.map e.symm K) p) (Eq (Submodule.map e p) K)
  -/
  constructor <;> rintro rfl
  · calc
      map e (map e.symm K) = comap e.symm (map e.symm K) := map_equiv_eq_comap_symm _ _
      _ = K := comap_map_eq_of_injective e.symm.injective _
  · calc
      map e.symm (map e p) = comap e (map e p) := (comap_equiv_eq_map_symm _ _).symm
      _ = p := comap_map_eq_of_injective e.injective _


theorem orderIsoMapComap_apply' (e : M ≃ₛₗ[τ₁₂] M₂) (p : Submodule R M) :
    orderIsoMapComap e p = comap e.symm p :=
  p.map_equiv_eq_comap_symm _


theorem orderIsoMapComap_symm_apply' (e : M ≃ₛₗ[τ₁₂] M₂) (p : Submodule R₂ M₂) :
    (orderIsoMapComap e).symm p = map e.symm p :=
  p.comap_equiv_eq_map_symm _


theorem inf_comap_le_comap_add (f₁ f₂ : M →ₛₗ[τ₁₂] M₂) :
    comap f₁ q ⊓ comap f₂ q ≤ comap (f₁ + f₂) q := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f₁ f₂ : LinearMap τ₁₂ M M₂
    ⊢ LE.le (Min.min (Submodule.comap f₁ q) (Submodule.comap f₂ q)) (Submodule.com …
  -/
  rw [SetLike.le_def]
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f₁ f₂ : LinearMap τ₁₂ M M₂
    ⊢ ∀ ⦃x : M⦄, Membership.mem (Min.min (Submodule.comap f₁ q) (Submodule.comap f …
  -/
  intro m h
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f₁ f₂ : LinearMap τ₁₂ M M₂
    m : M
    h : Membership.mem (Min.min (Submodule.comap f₁ q) (Submodule.comap f₂ q)) m
    ⊢ Membership.mem (Submodule.comap (HAdd.hAdd f₁ f₂) q) m
  -/
  change f₁ m + f₂ m ∈ q
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f₁ f₂ : LinearMap τ₁₂ M M₂
    m : M
    h : Membership.mem (Min.min (Submodule.comap f₁ q) (Submodule.comap f₂ q)) m
    ⊢ Membership.mem q (HAdd.hAdd (f₁ m) (f₂ m))
  -/
  change f₁ m ∈ q ∧ f₂ m ∈ q at h
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f₁ f₂ : LinearMap τ₁₂ M M₂
    m : M
    h : And (Membership.mem q (f₁ m)) (Membership.mem q (f₂ m))
    ⊢ Membership.mem q (HAdd.hAdd (f₁ m) (f₂ m))
  -/
  apply q.add_mem h.1 h.2
  /-
    🎉 no goals
  -/


theorem comap_le_comap_smul (fₗ : N →ₗ[R] N₂) (c : R) : comap fₗ qₗ ≤ comap (c • fₗ) qₗ := by
  /-
    R : Type u_1
    N : Type u_9
    N₂ : Type u_10
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid N
    inst✝² : AddCommMonoid N₂
    inst✝¹ : Module R N
    inst✝ : Module R N₂
    qₗ : Submodule R N₂
    fₗ : LinearMap (RingHom.id R) N N₂
    c : R
    ⊢ LE.le (Submodule.comap fₗ qₗ) (Submodule.comap (HSMul.hSMul c fₗ) qₗ)
  -/
  rw [SetLike.le_def]
  /-
    R : Type u_1
    N : Type u_9
    N₂ : Type u_10
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid N
    inst✝² : AddCommMonoid N₂
    inst✝¹ : Module R N
    inst✝ : Module R N₂
    qₗ : Submodule R N₂
    fₗ : LinearMap (RingHom.id R) N N₂
    c : R
    ⊢ ∀ ⦃x : N⦄, Membership.mem (Submodule.comap fₗ qₗ) x → Membership.mem (Submod …
  -/
  intro m h
  /-
    R : Type u_1
    N : Type u_9
    N₂ : Type u_10
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid N
    inst✝² : AddCommMonoid N₂
    inst✝¹ : Module R N
    inst✝ : Module R N₂
    qₗ : Submodule R N₂
    fₗ : LinearMap (RingHom.id R) N N₂
    c : R
    m : N
    h : Membership.mem (Submodule.comap fₗ qₗ) m
    ⊢ Membership.mem (Submodule.comap (HSMul.hSMul c fₗ) qₗ) m
  -/
  change c • fₗ m ∈ qₗ
  /-
    R : Type u_1
    N : Type u_9
    N₂ : Type u_10
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid N
    inst✝² : AddCommMonoid N₂
    inst✝¹ : Module R N
    inst✝ : Module R N₂
    qₗ : Submodule R N₂
    fₗ : LinearMap (RingHom.id R) N N₂
    c : R
    m : N
    h : Membership.mem (Submodule.comap fₗ qₗ) m
    ⊢ Membership.mem qₗ (HSMul.hSMul c (fₗ m))
  -/
  change fₗ m ∈ qₗ at h
  /-
    R : Type u_1
    N : Type u_9
    N₂ : Type u_10
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid N
    inst✝² : AddCommMonoid N₂
    inst✝¹ : Module R N
    inst✝ : Module R N₂
    qₗ : Submodule R N₂
    fₗ : LinearMap (RingHom.id R) N N₂
    c : R
    m : N
    h : Membership.mem qₗ (fₗ m)
    ⊢ Membership.mem qₗ (HSMul.hSMul c (fₗ m))
  -/
  apply qₗ.smul_mem _ h
  /-
    🎉 no goals
  -/


/-- Given modules `M`, `M₂` over a commutative ring, together with submodules `p ⊆ M`, `q ⊆ M₂`,
the set of maps $\{f ∈ Hom(M, M₂) | f(p) ⊆ q \}$ is a submodule of `Hom(M, M₂)`. -/
def compatibleMaps : Submodule R (N →ₗ[R] N₂) where
  carrier := { fₗ | pₗ ≤ comap fₗ qₗ }
  zero_mem' := by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      ⊢ Membership.mem { carrier := setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ) …
    -/
    change pₗ ≤ comap (0 : N →ₗ[R] N₂) qₗ
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      ⊢ LE.le pₗ (Submodule.comap 0 qₗ)
    -/
    rw [comap_zero]
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      f₁ f₂ : LinearMap (RingHom.id R) N N₂
      h₁ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₁
      h₂ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₂
      ⊢ Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) (HAdd.hAdd …
    -/
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      ⊢ LE.le pₗ Top.top
    -/
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      f₁ f₂ : LinearMap (RingHom.id R) N N₂
      h₁ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₁
      h₂ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₂
      ⊢ LE.le pₗ (Min.min (Submodule.comap f₁ qₗ) (Submodule.comap f₂ qₗ))
    -/
    exact le_top
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      f₁ f₂ : LinearMap (RingHom.id R) N N₂
      h₁ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₁
      h₂ : Membership.mem (setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ)) f₂
      ⊢ And (LE.le pₗ (Submodule.comap f₁ qₗ)) (LE.le pₗ (Submodule.comap f₂ qₗ))
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  add_mem' {f₁ f₂} h₁ h₂ := by
    apply le_trans _ (inf_comap_le_comap_add qₗ f₁ f₂)
    rw [le_inf_iff]
    exact ⟨h₁, h₂⟩
  smul_mem' c fₗ h := by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      c : R
      fₗ : LinearMap (RingHom.id R) N N₂
      h : Membership.mem { carrier := setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ q …
      ⊢ Membership.mem { carrier := setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ) …
    -/
    dsimp at h
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      N : Type u_9
      N₂ : Type u_10
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : CommSemiring R₂
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M
      inst✝⁶ : Module R₂ M₂
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N
      inst✝² : Module R N₂
      τ₁₂ : RingHom R R₂
      τ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair τ₁₂ τ₂₁
      inst✝ : RingHomInvPair τ₂₁ τ₁₂
      p : Submodule R M
      q : Submodule R₂ M₂
      pₗ : Submodule R N
      qₗ : Submodule R N₂
      c : R
      fₗ : LinearMap (RingHom.id R) N N₂
      h : LE.le pₗ (Submodule.comap fₗ qₗ)
      ⊢ Membership.mem { carrier := setOf fun fₗ => LE.le pₗ (Submodule.comap fₗ qₗ) …
    -/
    exact le_trans h (comap_le_comap_smul qₗ fₗ c)
    /-
      🎉 no goals
    -/


/-- A linear map between two modules restricts to a linear map from any submodule p of the
domain onto the image of that submodule.

This is the linear version of `AddMonoidHom.addSubmonoidMap` and `AddMonoidHom.addSubgroupMap`. -/
def submoduleMap (f : M →ₗ[R] M₁) (p : Submodule R M) : p →ₗ[R] p.map f :=
  f.restrict fun x hx ↦ Submodule.mem_map.mpr ⟨x, hx, rfl⟩


@[simp]
theorem submoduleMap_coe_apply (f : M →ₗ[R] M₁) {p : Submodule R M} (x : p) :
    ↑(f.submoduleMap p x) = f x := rfl


theorem submoduleMap_surjective (f : M →ₗ[R] M₁) (p : Submodule R M) :
    Function.Surjective (f.submoduleMap p) := f.toAddMonoidHom.addSubmonoidMap_surjective _


theorem map_codRestrict [RingHomSurjective σ₂₁] (p : Submodule R M) (f : M₂ →ₛₗ[σ₂₁] M) (h p') :
    Submodule.map (codRestrict p f h) p' = comap p.subtype (p'.map f) :=
                                  /-
                                    R : Type u_1
                                    R₂ : Type u_3
                                    M : Type u_5
                                    M₂ : Type u_7
                                    inst✝⁶ : Semiring R
                                    inst✝⁵ : AddCommMonoid M
                                    inst✝⁴ : Module R M
                                    inst✝³ : Semiring R₂
                                    inst✝² : AddCommMonoid M₂
                                    inst✝¹ : Module R₂ M₂
                                    σ₂₁ : RingHom R₂ R
                                    inst✝ : RingHomSurjective σ₂₁
                                    p : Submodule R M
                                    f : LinearMap σ₂₁ M₂ M
                                    h : ∀ (c : M₂), Membership.mem p (f c)
                                    p' : Submodule R₂ M₂
                                    x✝ : Subtype fun x => Membership.mem p x
                                    x : M
                                    hx : Membership.mem p x
                                    ⊢ Iff (Membership.mem (Submodule.map (LinearMap.codRestrict p f h) p') ⟨x, hx⟩ …
                                  -/
  Submodule.ext fun ⟨x, hx⟩ => by simp [Subtype.ext_iff_val]
                                  /-
                                    🎉 no goals
                                  -/


theorem comap_codRestrict (p : Submodule R M) (f : M₂ →ₛₗ[σ₂₁] M) (hf p') :
    Submodule.comap (codRestrict p f hf) p' = Submodule.comap f (map p.subtype p') :=
                                                           /-
                                                             R : Type u_1
                                                             R₂ : Type u_3
                                                             M : Type u_5
                                                             M₂ : Type u_7
                                                             inst✝⁵ : Semiring R
                                                             inst✝⁴ : AddCommMonoid M
                                                             inst✝³ : Module R M
                                                             inst✝² : Semiring R₂
                                                             inst✝¹ : AddCommMonoid M₂
                                                             inst✝ : Module R₂ M₂
                                                             σ₂₁ : RingHom R₂ R
                                                             p : Submodule R M
                                                             f : LinearMap σ₂₁ M₂ M
                                                             hf : ∀ (c : M₂), Membership.mem p (f c)
                                                             p' : Submodule R (Subtype fun x => Membership.mem p x)
                                                             x : M₂
                                                             ⊢ Membership.mem (Submodule.comap f (Submodule.map p.subtype p')) x → Membersh …
                                                           -/
  Submodule.ext fun x => ⟨fun h => ⟨⟨_, hf x⟩, h, rfl⟩, by rintro ⟨⟨_, _⟩, h, ⟨⟩⟩; exact h⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem map_eq_comap {p : Submodule R M} :
    (p.map (e : M →ₛₗ[σ₁₂] M₂) : Submodule R₂ M₂) = p.comap (e.symm : M₂ →ₛₗ[σ₂₁] M) :=
                              /-
                                R : Type u_1
                                R₂ : Type u_3
                                M : Type u_5
                                M₂ : Type u_7
                                inst✝³ : Semiring R
                                inst✝² : Semiring R₂
                                inst✝¹ : AddCommMonoid M
                                inst✝ : AddCommMonoid M₂
                                module_M : Module R M
                                module_M₂ : Module R₂ M₂
                                σ₁₂ : RingHom R R₂
                                σ₂₁ : RingHom R₂ R
                                re₁₂ : RingHomInvPair σ₁₂ σ₂₁
                                re₂₁ : RingHomInvPair σ₂₁ σ₁₂
                                e : LinearEquiv σ₁₂ M M₂
                                p : Submodule R M
                                ⊢ Eq ↑(Submodule.map (↑e) p) ↑(Submodule.comap (↑e.symm) p)
                              -/
  SetLike.coe_injective <| by simp [e.image_eq_preimage]
                              /-
                                🎉 no goals
                              -/


/-- A linear equivalence of two modules restricts to a linear equivalence from any submodule
`p` of the domain onto the image of that submodule.

This is the linear version of `AddEquiv.submonoidMap` and `AddEquiv.subgroupMap`.

This is `LinearEquiv.ofSubmodule'` but with `map` on the right instead of `comap` on the left. -/
def submoduleMap (p : Submodule R M) : p ≃ₛₗ[σ₁₂] ↥(p.map (e : M →ₛₗ[σ₁₂] M₂) : Submodule R₂ M₂) :=
  { ((e : M →ₛₗ[σ₁₂] M₂).domRestrict p).codRestrict (p.map (e : M →ₛₗ[σ₁₂] M₂)) fun x =>
      ⟨x, by
        simp only [LinearMap.domRestrict_apply, eq_self_iff_true, and_true, SetLike.coe_mem,
          SetLike.mem_coe]⟩ with
    invFun := fun y =>
      ⟨(e.symm : M₂ →ₛₗ[σ₂₁] M) y, by
        /-
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          M : Type u_5
          M₁ : Type u_6
          M₂ : Type u_7
          M₃ : Type u_8
          inst✝³ : Semiring R
          inst✝² : Semiring R₂
          inst✝¹ : AddCommMonoid M
          inst✝ : AddCommMonoid M₂
          module_M : Module R M
          module_M₂ : Module R₂ M₂
          σ₁₂ : RingHom R R₂
          σ₂₁ : RingHom R₂ R
          re₁₂ : RingHomInvPair σ₁₂ σ₂₁
          re₂₁ : RingHomInvPair σ₂₁ σ₁₂
          e : LinearEquiv σ₁₂ M M₂
          p : Submodule R M
          y : Subtype fun x => Membership.mem (Submodule.map (↑e) p) x
          ⊢ Membership.mem p (↑e.symm ↑y)
        -/
        rcases y with ⟨y', hy⟩
        /-
          case mk
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          M : Type u_5
          M₁ : Type u_6
          M₂ : Type u_7
          M₃ : Type u_8
          inst✝³ : Semiring R
          inst✝² : Semiring R₂
          inst✝¹ : AddCommMonoid M
          inst✝ : AddCommMonoid M₂
          module_M : Module R M
          module_M₂ : Module R₂ M₂
          σ₁₂ : RingHom R R₂
          σ₂₁ : RingHom R₂ R
          re₁₂ : RingHomInvPair σ₁₂ σ₂₁
          re₂₁ : RingHomInvPair σ₂₁ σ₁₂
          e : LinearEquiv σ₁₂ M M₂
          p : Submodule R M
          y' : M₂
          hy : Membership.mem (Submodule.map (↑e) p) y'
          ⊢ Membership.mem p (↑e.symm ↑⟨y', hy⟩)
        -/
        rw [Submodule.mem_map] at hy
        /-
          case mk
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          M : Type u_5
          M₁ : Type u_6
          M₂ : Type u_7
          M₃ : Type u_8
          inst✝³ : Semiring R
          inst✝² : Semiring R₂
          inst✝¹ : AddCommMonoid M
          inst✝ : AddCommMonoid M₂
          module_M : Module R M
          module_M₂ : Module R₂ M₂
          σ₁₂ : RingHom R R₂
          σ₂₁ : RingHom R₂ R
          re₁₂ : RingHomInvPair σ₁₂ σ₂₁
          re₂₁ : RingHomInvPair σ₂₁ σ₁₂
          e : LinearEquiv σ₁₂ M M₂
          p : Submodule R M
          y' : M₂
          hy✝ : Membership.mem (Submodule.map (↑e) p) y'
          hy : Exists fun y => And (Membership.mem p y) (Eq (↑e y) y')
          ⊢ Membership.mem p (↑e.symm ↑⟨y', hy✝⟩)
        -/
        rcases hy with ⟨x, hx, hxy⟩
        /-
          case mk.intro.intro
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          M : Type u_5
          M₁ : Type u_6
          M₂ : Type u_7
          M₃ : Type u_8
          inst✝³ : Semiring R
          inst✝² : Semiring R₂
          inst✝¹ : AddCommMonoid M
          inst✝ : AddCommMonoid M₂
          module_M : Module R M
          module_M₂ : Module R₂ M₂
          σ₁₂ : RingHom R R₂
          σ₂₁ : RingHom R₂ R
          re₁₂ : RingHomInvPair σ₁₂ σ₂₁
          re₂₁ : RingHomInvPair σ₂₁ σ₁₂
          e : LinearEquiv σ₁₂ M M₂
          p : Submodule R M
          y' : M₂
          hy : Membership.mem (Submodule.map (↑e) p) y'
          x : M
          hx : Membership.mem p x
          hxy : Eq (↑e x) y'
          ⊢ Membership.mem p (↑e.symm ↑⟨y', hy⟩)
        -/
        subst hxy
        /-
          case mk.intro.intro
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          M : Type u_5
          M₁ : Type u_6
          M₂ : Type u_7
          M₃ : Type u_8
          inst✝³ : Semiring R
          inst✝² : Semiring R₂
          inst✝¹ : AddCommMonoid M
          inst✝ : AddCommMonoid M₂
          module_M : Module R M
          module_M₂ : Module R₂ M₂
          σ₁₂ : RingHom R R₂
          σ₂₁ : RingHom R₂ R
          re₁₂ : RingHomInvPair σ₁₂ σ₂₁
          re₂₁ : RingHomInvPair σ₂₁ σ₁₂
          e : LinearEquiv σ₁₂ M M₂
          p : Submodule R M
          x : M
          hx : Membership.mem p x
          hy : Membership.mem (Submodule.map (↑e) p) (↑e x)
          ⊢ Membership.mem p (↑e.symm ↑⟨↑e x, hy⟩)
        -/
        simp only [symm_apply_apply, Submodule.coe_mk, coe_coe, hx]⟩
        /-
          🎉 no goals
        -/
    left_inv := fun x => by
      simp only [LinearMap.domRestrict_apply, LinearMap.codRestrict_apply, LinearMap.toFun_eq_coe,
        LinearEquiv.coe_coe, LinearEquiv.symm_apply_apply, SetLike.eta]
    right_inv := fun y => by
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝³ : Semiring R
        inst✝² : Semiring R₂
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid M₂
        module_M : Module R M
        module_M₂ : Module R₂ M₂
        σ₁₂ : RingHom R R₂
        σ₂₁ : RingHom R₂ R
        re₁₂ : RingHomInvPair σ₁₂ σ₂₁
        re₂₁ : RingHomInvPair σ₂₁ σ₁₂
        e : LinearEquiv σ₁₂ M M₂
        p : Submodule R M
        y : Subtype fun x => Membership.mem (Submodule.map (↑e) p) x
        ⊢ Eq (__src✝.toFun ((fun y => ⟨↑e.symm ↑y, ⋯⟩) y)) y
      -/
      apply SetCoe.ext
      simp only [LinearMap.domRestrict_apply, LinearMap.codRestrict_apply, LinearMap.toFun_eq_coe,
        LinearEquiv.coe_coe, LinearEquiv.apply_symm_apply] }


@[simp]
theorem submoduleMap_apply (p : Submodule R M) (x : p) : ↑(e.submoduleMap p x) = e x :=
  rfl


@[simp]
theorem submoduleMap_symm_apply (p : Submodule R M)
    (x : (p.map (e : M →ₛₗ[σ₁₂] M₂) : Submodule R₂ M₂)) : ↑((e.submoduleMap p).symm x) = e.symm x :=
  rfl


