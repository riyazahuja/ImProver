/-- The setoid whose quotient is the projectivization of `V`. -/
def projectivizationSetoid : Setoid { v : V // v ≠ 0 } :=
  (MulAction.orbitRel Kˣ V).comap (↑)


/-- The projectivization of the `K`-vector space `V`.
The notation `ℙ K V` is preferred. -/
def Projectivization := Quotient (projectivizationSetoid K V)


/-- We define notations `ℙ K V` for the projectivization of the `K`-vector space `V`. -/
scoped[LinearAlgebra.Projectivization] notation "ℙ" => Projectivization


/-- Construct an element of the projectivization from a nonzero vector. -/
def mk (v : V) (hv : v ≠ 0) : ℙ K V :=
  Quotient.mk'' ⟨v, hv⟩


/-- A variant of `Projectivization.mk` in terms of a subtype. `mk` is preferred. -/
def mk' (v : { v : V // v ≠ 0 }) : ℙ K V :=
  Quotient.mk'' v


@[simp]
theorem mk'_eq_mk (v : { v : V // v ≠ 0 }) : mk' K v = mk K ↑v v.2 := rfl


instance [Nontrivial V] : Nonempty (ℙ K V) :=
  let ⟨v, hv⟩ := exists_ne (0 : V)
  ⟨mk K v hv⟩


/-- A function on non-zero vectors which is independent of scale, descends to a function on the
projectivization. -/
protected def lift {α : Type*} (f : { v : V // v ≠ 0 } → α)
    (hf : ∀ (a b : { v : V // v ≠ 0 }) (t : K), a = t • (b : V) → f a = f b)
    (x : ℙ K V) : α :=
                      /-
                        K : Type u_1
                        V : Type u_2
                        inst✝² : DivisionRing K
                        inst✝¹ : AddCommGroup V
                        inst✝ : Module K V
                        α : Type u_3
                        f : (Subtype fun v => Ne v 0) → α
                        hf : ∀ (a b : Subtype fun v => Ne v 0) (t : K), Eq (↑a) (HSMul.hSMul t ↑b) → E …
                        x : Projectivization K V
                        ⊢ ∀ (a b : Subtype fun v => Ne v 0), HasEquiv.Equiv a b → Eq (f a) (f b)
                      -/
  Quotient.lift f (by rintro ⟨-, hv⟩ ⟨w, hw⟩ ⟨⟨t, -⟩, rfl⟩; exact hf ⟨_, hv⟩ ⟨w, hw⟩ t rfl) x
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
protected lemma lift_mk {α : Type*} (f : { v : V // v ≠ 0 } → α)
    (hf : ∀ (a b : { v : V // v ≠ 0 }) (t : K), a = t • (b : V) → f a = f b)
    (v : V) (hv : v ≠ 0) :
    Projectivization.lift f hf (mk K v hv) = f ⟨v, hv⟩ :=
  rfl


/-- Choose a representative of `v : Projectivization K V` in `V`. -/
protected noncomputable def rep (v : ℙ K V) : V :=
  v.out


theorem rep_nonzero (v : ℙ K V) : v.rep ≠ 0 :=
  v.out.2


@[simp]
theorem mk_rep (v : ℙ K V) : mk K v.rep v.rep_nonzero = v := Quotient.out_eq' _


/-- Consider an element of the projectivization as a submodule of `V`. -/
protected def submodule (v : ℙ K V) : Submodule K V :=
  (Quotient.liftOn' v fun v => K ∙ (v : V)) <| by
    /-
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : Projectivization K V
      ⊢ ∀ (a b : Subtype fun v => Ne v 0), (projectivizationSetoid K V) a b → Eq (Su …
    -/
    rintro ⟨a, ha⟩ ⟨b, hb⟩ ⟨x, rfl : x • b = a⟩
    /-
      case mk.mk.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : Projectivization K V
      b : V
      hb : Ne b 0
      x : Units K
      ha : Ne (HSMul.hSMul x b) 0
      ⊢ Eq (Submodule.span K (Singleton.singleton ↑⟨HSMul.hSMul x b, ha⟩)) (Submodul …
    -/
    exact Submodule.span_singleton_group_smul_eq _ x _
    /-
      🎉 no goals
    -/


theorem mk_eq_mk_iff (v w : V) (hv : v ≠ 0) (hw : w ≠ 0) :
    mk K v hv = mk K w hw ↔ ∃ a : Kˣ, a • w = v :=
  Quotient.eq''


/-- Two nonzero vectors go to the same point in projective space if and only if one is
a scalar multiple of the other. -/
theorem mk_eq_mk_iff' (v w : V) (hv : v ≠ 0) (hw : w ≠ 0) :
    mk K v hv = mk K w hw ↔ ∃ a : K, a • w = v := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v w : V
    hv : Ne v 0
    hw : Ne w 0
    ⊢ Iff (Eq (Projectivization.mk K v hv) (Projectivization.mk K w hw)) (Exists f …
  -/
  rw [mk_eq_mk_iff K v w hv hw]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v w : V
    hv : Ne v 0
    hw : Ne w 0
    ⊢ Iff (Exists fun a => Eq (HSMul.hSMul a w) v) (Exists fun a => Eq (HSMul.hSMu …
  -/
  constructor
    /-
      case mp
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v w : V
      hv : Ne v 0
      hw : Ne w 0
      ⊢ (Exists fun a => Eq (HSMul.hSMul a w) v) → Exists fun a => Eq (HSMul.hSMul a …
    -/
  · rintro ⟨a, ha⟩
    /-
      case mp.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v w : V
      hv : Ne v 0
      hw : Ne w 0
      a : Units K
      ha : Eq (HSMul.hSMul a w) v
      ⊢ Exists fun a => Eq (HSMul.hSMul a w) v
    -/
    exact ⟨a, ha⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v w : V
      hv : Ne v 0
      hw : Ne w 0
      ⊢ (Exists fun a => Eq (HSMul.hSMul a w) v) → Exists fun a => Eq (HSMul.hSMul a …
    -/
  · rintro ⟨a, ha⟩
    /-
      case mpr.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v w : V
      hv : Ne v 0
      hw : Ne w 0
      a : K
      ha : Eq (HSMul.hSMul a w) v
      ⊢ Exists fun a => Eq (HSMul.hSMul a w) v
    -/
    refine ⟨Units.mk0 a fun c => hv.symm ?_, ha⟩
    /-
      case mpr.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v w : V
      hv : Ne v 0
      hw : Ne w 0
      a : K
      ha : Eq (HSMul.hSMul a w) v
      c : Eq a 0
      ⊢ Eq 0 v
    -/
    rwa [c, zero_smul] at ha
    /-
      🎉 no goals
    -/


theorem exists_smul_eq_mk_rep (v : V) (hv : v ≠ 0) : ∃ a : Kˣ, a • v = (mk K v hv).rep :=
  (mk_eq_mk_iff K _ _ (rep_nonzero _) hv).1 (mk_rep _)


/-- An induction principle for `Projectivization`. Use as `induction v`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
theorem ind {P : ℙ K V → Prop} (h : ∀ (v : V) (h : v ≠ 0), P (mk K v h)) : ∀ p, P p :=
  Quotient.ind' <| Subtype.rec <| h


@[simp]
theorem submodule_mk (v : V) (hv : v ≠ 0) : (mk K v hv).submodule = K ∙ v :=
  rfl


theorem submodule_eq (v : ℙ K V) : v.submodule = K ∙ v.rep := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ Eq v.submodule (Submodule.span K (Singleton.singleton v.rep))
  -/
  conv_lhs => rw [← v.mk_rep]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ Eq (Projectivization.mk K v.rep ⋯).submodule (Submodule.span K (Singleton.si …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem finrank_submodule (v : ℙ K V) : finrank K v.submodule = 1 := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem v.submodule x)) 1
  -/
  rw [submodule_eq]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.span K (Sin …
  -/
  exact finrank_span_singleton v.rep_nonzero
  /-
    🎉 no goals
  -/


instance (v : ℙ K V) : FiniteDimensional K v.submodule := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem v.submodule x)
  -/
  rw [← v.mk_rep]
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Projectivization.mk K  …
  -/
  change FiniteDimensional K (K ∙ v.rep)
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Submodule.span K (Sing …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem submodule_injective :
    Function.Injective (Projectivization.submodule : ℙ K V → Submodule K V) := fun u v h ↦ by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    u v : Projectivization K V
    h : Eq u.submodule v.submodule
    ⊢ Eq u v
  -/
  induction' u using ind with u hu
  /-
    case h
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : Projectivization K V
    u : V
    hu : Ne u 0
    h : Eq (Projectivization.mk K u hu).submodule v.submodule
    ⊢ Eq (Projectivization.mk K u hu) v
  -/
  induction' v using ind with v hv
  /-
    case h.h
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    h : Eq (Projectivization.mk K u hu).submodule (Projectivization.mk K v hv).sub …
    ⊢ Eq (Projectivization.mk K u hu) (Projectivization.mk K v hv)
  -/
  rw [submodule_mk, submodule_mk, Submodule.span_singleton_eq_span_singleton] at h
  /-
    case h.h
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    h : Exists fun z => Eq (HSMul.hSMul z u) v
    ⊢ Eq (Projectivization.mk K u hu) (Projectivization.mk K v hv)
  -/
  exact ((mk_eq_mk_iff K v u hv hu).2 h).symm
  /-
    🎉 no goals
  -/


/-- The equivalence between the projectivization and the
collection of subspaces of dimension 1. -/
noncomputable def equivSubmodule : ℙ K V ≃ { H : Submodule K V // finrank K H = 1 } :=
  (Equiv.ofInjective _ submodule_injective).trans <| .subtypeEquiv (.refl _) fun H ↦ by
    /-
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      ⊢ Iff (Membership.mem (Set.range Projectivization.submodule) H) (Eq (Module.fi …
    -/
    refine ⟨fun ⟨v, hv⟩ ↦ hv ▸ v.finrank_submodule, fun h ↦ ?_⟩
    /-
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      ⊢ Membership.mem (Set.range Projectivization.submodule) H
    -/
    rcases finrank_eq_one_iff'.1 h with ⟨v : H, hv₀, hv : ∀ w : H, _⟩
    /-
      case intro.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      v : Subtype fun x => Membership.mem H x
      hv₀ : Ne v 0
      hv : ∀ (w : Subtype fun x => Membership.mem H x), Exists fun c => Eq (HSMul.hS …
      ⊢ Membership.mem (Set.range Projectivization.submodule) H
    -/
    use mk K (v : V) (Subtype.coe_injective.ne hv₀)
    /-
      case h
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      v : Subtype fun x => Membership.mem H x
      hv₀ : Ne v 0
      hv : ∀ (w : Subtype fun x => Membership.mem H x), Exists fun c => Eq (HSMul.hS …
      ⊢ Eq (Projectivization.mk K ↑v ⋯).submodule H
    -/
    rw [submodule_mk, SetLike.ext'_iff, Submodule.span_singleton_eq_range]
    /-
      case h
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      v : Subtype fun x => Membership.mem H x
      hv₀ : Ne v 0
      hv : ∀ (w : Subtype fun x => Membership.mem H x), Exists fun c => Eq (HSMul.hS …
      ⊢ Eq (Set.range fun x => HSMul.hSMul x ↑v) ↑H
    -/
    refine (Set.range_subset_iff.2 fun _ ↦ H.smul_mem _ v.2).antisymm fun x hx ↦ ?_
    /-
      case h
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      v : Subtype fun x => Membership.mem H x
      hv₀ : Ne v 0
      hv : ∀ (w : Subtype fun x => Membership.mem H x), Exists fun c => Eq (HSMul.hS …
      x : V
      hx : Membership.mem (↑H) x
      ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑v) x
    -/
    rcases hv ⟨x, hx⟩ with ⟨c, hc⟩
    /-
      case h.intro
      K : Type u_1
      V : Type u_2
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      H : Submodule K V
      h : Eq (Module.finrank K (Subtype fun x => Membership.mem ((Equiv.refl (Submod …
      v : Subtype fun x => Membership.mem H x
      hv₀ : Ne v 0
      hv : ∀ (w : Subtype fun x => Membership.mem H x), Exists fun c => Eq (HSMul.hS …
      x : V
      hx : Membership.mem (↑H) x
      c : K
      hc : Eq (HSMul.hSMul c v) ⟨x, hx⟩
      ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑v) x
    -/
    exact ⟨c, congr_arg Subtype.val hc⟩
    /-
      🎉 no goals
    -/


/-- Construct an element of the projectivization from a subspace of dimension 1. -/
noncomputable def mk'' (H : Submodule K V) (h : finrank K H = 1) : ℙ K V :=
  (equivSubmodule K V).symm ⟨H, h⟩


@[simp]
theorem submodule_mk'' (H : Submodule K V) (h : finrank K H = 1) : (mk'' H h).submodule = H :=
  congr_arg Subtype.val <| (equivSubmodule K V).apply_symm_apply ⟨H, h⟩


@[simp]
theorem mk''_submodule (v : ℙ K V) : mk'' v.submodule v.finrank_submodule = v :=
  (equivSubmodule K V).symm_apply_apply v


/-- An injective semilinear map of vector spaces induces a map on projective spaces. -/
def map {σ : K →+* L} (f : V →ₛₗ[σ] W) (hf : Function.Injective f) : ℙ K V → ℙ L W :=
                                                     /-
                                                       K : Type u_1
                                                       V : Type u_2
                                                       inst✝⁵ : DivisionRing K
                                                       inst✝⁴ : AddCommGroup V
                                                       inst✝³ : Module K V
                                                       L : Type u_3
                                                       W : Type u_4
                                                       inst✝² : DivisionRing L
                                                       inst✝¹ : AddCommGroup W
                                                       inst✝ : Module L W
                                                       σ : RingHom K L
                                                       f : LinearMap σ V W
                                                       hf : Function.Injective ⇑f
                                                       v : Subtype fun v => Ne v 0
                                                       c : Eq (f ↑v) 0
                                                       ⊢ Eq (f ↑v) (f 0)
                                                     -/
  Quotient.map' (fun v => ⟨f v, fun c => v.2 (hf (by simp [c]))⟩)
                                                     /-
                                                       🎉 no goals
                                                     -/
    (by
      /-
        K : Type u_1
        V : Type u_2
        inst✝⁵ : DivisionRing K
        inst✝⁴ : AddCommGroup V
        inst✝³ : Module K V
        L : Type u_3
        W : Type u_4
        inst✝² : DivisionRing L
        inst✝¹ : AddCommGroup W
        inst✝ : Module L W
        σ : RingHom K L
        f : LinearMap σ V W
        hf : Function.Injective ⇑f
        ⊢ ∀ (a b : Subtype fun v => Ne v 0), (projectivizationSetoid K V) a b → (proje …
      -/
      rintro ⟨u, hu⟩ ⟨v, hv⟩ ⟨a, ha⟩
      /-
        case mk.mk.intro
        K : Type u_1
        V : Type u_2
        inst✝⁵ : DivisionRing K
        inst✝⁴ : AddCommGroup V
        inst✝³ : Module K V
        L : Type u_3
        W : Type u_4
        inst✝² : DivisionRing L
        inst✝¹ : AddCommGroup W
        inst✝ : Module L W
        σ : RingHom K L
        f : LinearMap σ V W
        hf : Function.Injective ⇑f
        u : V
        hu : Ne u 0
        v : V
        hv : Ne v 0
        a : Units K
        ha : Eq ((fun m => HSMul.hSMul m ↑⟨v, hv⟩) a) ↑⟨u, hu⟩
        ⊢ (projectivizationSetoid L W) ((fun v => ⟨f ↑v, ⋯⟩) ⟨u, hu⟩) ((fun v => ⟨f ↑v …
      -/
      use Units.map σ.toMonoidHom a
      /-
        case h
        K : Type u_1
        V : Type u_2
        inst✝⁵ : DivisionRing K
        inst✝⁴ : AddCommGroup V
        inst✝³ : Module K V
        L : Type u_3
        W : Type u_4
        inst✝² : DivisionRing L
        inst✝¹ : AddCommGroup W
        inst✝ : Module L W
        σ : RingHom K L
        f : LinearMap σ V W
        hf : Function.Injective ⇑f
        u : V
        hu : Ne u 0
        v : V
        hv : Ne v 0
        a : Units K
        ha : Eq ((fun m => HSMul.hSMul m ↑⟨v, hv⟩) a) ↑⟨u, hu⟩
        ⊢ Eq ((fun m => HSMul.hSMul m ↑((fun v => ⟨f ↑v, ⋯⟩) ⟨v, hv⟩)) ((Units.map ↑σ) …
      -/
      dsimp at ha ⊢
      /-
        case h
        K : Type u_1
        V : Type u_2
        inst✝⁵ : DivisionRing K
        inst✝⁴ : AddCommGroup V
        inst✝³ : Module K V
        L : Type u_3
        W : Type u_4
        inst✝² : DivisionRing L
        inst✝¹ : AddCommGroup W
        inst✝ : Module L W
        σ : RingHom K L
        f : LinearMap σ V W
        hf : Function.Injective ⇑f
        u : V
        hu : Ne u 0
        v : V
        hv : Ne v 0
        a : Units K
        ha : Eq (HSMul.hSMul a v) u
        ⊢ Eq (HSMul.hSMul ((Units.map ↑σ) a) (f v)) (f u)
      -/
      erw [← f.map_smulₛₗ, ha])
      /-
        🎉 no goals
      -/


theorem map_mk {σ : K →+* L} (f : V →ₛₗ[σ] W) (hf : Function.Injective f) (v : V) (hv : v ≠ 0) :
    map f hf (mk K v hv) = mk L (f v) (map_zero f ▸ hf.ne hv) :=
  rfl


/-- Mapping with respect to a semilinear map over an isomorphism of fields yields
an injective map on projective spaces. -/
theorem map_injective {σ : K →+* L} {τ : L →+* K} [RingHomInvPair σ τ] (f : V →ₛₗ[σ] W)
    (hf : Function.Injective f) : Function.Injective (map f hf) := fun u v h ↦ by
  /-
    K : Type u_1
    V : Type u_2
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝³ : DivisionRing L
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    σ : RingHom K L
    τ : RingHom L K
    inst✝ : RingHomInvPair σ τ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    u v : Projectivization K V
    h : Eq (Projectivization.map f hf u) (Projectivization.map f hf v)
    ⊢ Eq u v
  -/
  induction' u using ind with u hu; induction' v using ind with v hv
  /-
    case h.h
    K : Type u_1
    V : Type u_2
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝³ : DivisionRing L
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    σ : RingHom K L
    τ : RingHom L K
    inst✝ : RingHomInvPair σ τ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    h : Eq (Projectivization.map f hf (Projectivization.mk K u hu)) (Projectivizat …
    ⊢ Eq (Projectivization.mk K u hu) (Projectivization.mk K v hv)
  -/
  simp only [map_mk, mk_eq_mk_iff'] at h ⊢
  /-
    case h.h
    K : Type u_1
    V : Type u_2
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝³ : DivisionRing L
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    σ : RingHom K L
    τ : RingHom L K
    inst✝ : RingHomInvPair σ τ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    h : Exists fun a => Eq (HSMul.hSMul a (f v)) (f u)
    ⊢ Exists fun a => Eq (HSMul.hSMul a v) u
  -/
  rcases h with ⟨a, ha⟩
  /-
    case h.h.intro
    K : Type u_1
    V : Type u_2
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝³ : DivisionRing L
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    σ : RingHom K L
    τ : RingHom L K
    inst✝ : RingHomInvPair σ τ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    a : L
    ha : Eq (HSMul.hSMul a (f v)) (f u)
    ⊢ Exists fun a => Eq (HSMul.hSMul a v) u
  -/
  refine ⟨τ a, hf ?_⟩
  /-
    case h.h.intro
    K : Type u_1
    V : Type u_2
    inst✝⁶ : DivisionRing K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝³ : DivisionRing L
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    σ : RingHom K L
    τ : RingHom L K
    inst✝ : RingHomInvPair σ τ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    u : V
    hu : Ne u 0
    v : V
    hv : Ne v 0
    a : L
    ha : Eq (HSMul.hSMul a (f v)) (f u)
    ⊢ Eq (f (HSMul.hSMul (τ a) v)) (f u)
  -/
  rwa [f.map_smulₛₗ, RingHomInvPair.comp_apply_eq₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_id : map (LinearMap.id : V →ₗ[K] V) (LinearEquiv.refl K V).injective = id := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Eq (Projectivization.map LinearMap.id ⋯) id
  -/
  ext ⟨v⟩
  /-
    case h.mk
    K : Type u_1
    V : Type u_2
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x✝ : Projectivization K V
    v : Subtype fun v => Ne v 0
    ⊢ Eq (Projectivization.map LinearMap.id ⋯ (Quot.mk (⇑(projectivizationSetoid K …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: removed `@[simp]` because of unusable `hg.comp hf` in the LHS

theorem map_comp {F U : Type*} [Field F] [AddCommGroup U] [Module F U] {σ : K →+* L} {τ : L →+* F}
    {γ : K →+* F} [RingHomCompTriple σ τ γ] (f : V →ₛₗ[σ] W) (hf : Function.Injective f)
    (g : W →ₛₗ[τ] U) (hg : Function.Injective g) :
    map (g.comp f) (hg.comp hf) = map g hg ∘ map f hf := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝⁹ : DivisionRing K
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝⁶ : DivisionRing L
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module L W
    F : Type u_5
    U : Type u_6
    inst✝³ : Field F
    inst✝² : AddCommGroup U
    inst✝¹ : Module F U
    σ : RingHom K L
    τ : RingHom L F
    γ : RingHom K F
    inst✝ : RingHomCompTriple σ τ γ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    g : LinearMap τ W U
    hg : Function.Injective ⇑g
    ⊢ Eq (Projectivization.map (g.comp f) ⋯) (Function.comp (Projectivization.map  …
  -/
  ext ⟨v⟩
  /-
    case h.mk
    K : Type u_1
    V : Type u_2
    inst✝⁹ : DivisionRing K
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module K V
    L : Type u_3
    W : Type u_4
    inst✝⁶ : DivisionRing L
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module L W
    F : Type u_5
    U : Type u_6
    inst✝³ : Field F
    inst✝² : AddCommGroup U
    inst✝¹ : Module F U
    σ : RingHom K L
    τ : RingHom L F
    γ : RingHom K F
    inst✝ : RingHomCompTriple σ τ γ
    f : LinearMap σ V W
    hf : Function.Injective ⇑f
    g : LinearMap τ W U
    hg : Function.Injective ⇑g
    x✝ : Projectivization K V
    v : Subtype fun v => Ne v 0
    ⊢ Eq (Projectivization.map (g.comp f) ⋯ (Quot.mk (⇑(projectivizationSetoid K V …
  -/
  rfl
  /-
    🎉 no goals
  -/


