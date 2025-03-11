/--
A module `M` over a commutative ring `R` is *faithfully flat* if it is flat and,
for all `R`-module homomorphism `f : N → N'` such that `id ⊗ f = 0`, we have `f = 0`.
-/
@[mk_iff] class FaithfullyFlat extends Module.Flat R M : Prop where
  submodule_ne_top : ∀ ⦃m : Ideal R⦄ (_ : Ideal.IsMaximal m), m • (⊤ : Submodule R M) ≠ ⊤


instance self : FaithfullyFlat R R where
  submodule_ne_top m h r := Ideal.eq_top_iff_one _ |>.not.1 h.ne_top <| by
    /-
      R : Type u
      M : Type v
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      m : Ideal R
      h : m.IsMaximal
      r : Eq (HSMul.hSMul m Top.top) Top.top
      ⊢ Membership.mem m 1
    -/
    simpa using show 1 ∈ (m • ⊤ : Ideal R) from r.symm ▸ ⟨⟩
    /-
      🎉 no goals
    -/


lemma iff_flat_and_proper_ideal :
    FaithfullyFlat R M ↔
    (Flat R M ∧ ∀ (I : Ideal R), I ≠ ⊤ → I • (⊤ : Submodule R M) ≠ ⊤) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.FaithfullyFlat R M) (And (Module.Flat R M) (∀ (I : Ideal R), Ne  …
  -/
  rw [faithfullyFlat_iff]
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (And (Module.Flat R M) (∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m …
  -/
  refine ⟨fun ⟨flat, h⟩ => ⟨flat, fun I hI r => ?_⟩, fun h => ⟨h.1, fun m hm => h.2 _ hm.ne_top⟩⟩
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m T …
    flat : Module.Flat R M
    h : ∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m Top.top) Top.top
    I : Ideal R
    hI : Ne I Top.top
    r : Eq (HSMul.hSMul I Top.top) Top.top
    ⊢ False
  -/
  obtain ⟨m, hm, le⟩ := I.exists_le_maximal hI
  /-
    case intro.intro
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m T …
    flat : Module.Flat R M
    h : ∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m Top.top) Top.top
    I : Ideal R
    hI : Ne I Top.top
    r : Eq (HSMul.hSMul I Top.top) Top.top
    m : Ideal R
    hm : m.IsMaximal
    le : LE.le I m
    ⊢ False
  -/
  exact h hm <| eq_top_iff.2 <| show ⊤ ≤ m • ⊤ from r ▸ Submodule.smul_mono le (by simp [r])
  /-
    🎉 no goals
  -/


lemma iff_flat_and_ideal_smul_eq_top :
    FaithfullyFlat R M ↔
    (Flat R M ∧ ∀ (I : Ideal R), I • (⊤ : Submodule R M) = ⊤ → I = ⊤) :=
  iff_flat_and_proper_ideal R M |>.trans <| and_congr_right_iff.2 fun _ => iff_of_eq <|
                                             /-
                                               R : Type u
                                               M : Type v
                                               inst✝² : CommRing R
                                               inst✝¹ : AddCommGroup M
                                               inst✝ : Module R M
                                               x✝ : Module.Flat R M
                                               I : Ideal R
                                               ⊢ Iff (Ne I Top.top → Ne (HSMul.hSMul I Top.top) Top.top) (Eq (HSMul.hSMul I T …
                                             -/
    forall_congr fun I => eq_iff_iff.2 <| by tauto
                                             /-
                                               🎉 no goals
                                             -/


instance rTensor_nontrivial
    [fl: FaithfullyFlat R M] (N : Type*) [AddCommGroup N] [Module R N] [Nontrivial N] :
    Nontrivial (N ⊗[R] M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    fl : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Nontrivial N
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  obtain ⟨n, hn⟩ := nontrivial_iff_exists_ne (0 : N) |>.1 inferInstance
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    fl : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Nontrivial N
    n : N
    hn : Ne n 0
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  let I := (Submodule.span R {n}).annihilator
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    fl : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Nontrivial N
    n : N
    hn : Ne n 0
    I : Ideal R := (Submodule.span R (Singleton.singleton n)).annihilator
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  by_cases I_ne_top : I = ⊤
    /-
      case pos
      R : Type u
      M : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      fl : Module.FaithfullyFlat R M
      N : Type u_1
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : Nontrivial N
      n : N
      hn : Ne n 0
      I : Ideal R := (Submodule.span R (Singleton.singleton n)).annihilator
      I_ne_top : Eq I Top.top
      ⊢ Nontrivial (TensorProduct R N M)
    -/
  · rw [Ideal.eq_top_iff_one, Submodule.mem_annihilator_span_singleton, one_smul] at I_ne_top
    /-
      case pos
      R : Type u
      M : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      fl : Module.FaithfullyFlat R M
      N : Type u_1
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      inst✝ : Nontrivial N
      n : N
      hn : Ne n 0
      I : Ideal R := (Submodule.span R (Singleton.singleton n)).annihilator
      I_ne_top : Eq n 0
      ⊢ Nontrivial (TensorProduct R N M)
    -/
    contradiction
    /-
      🎉 no goals
    -/
  let inc : R ⧸ I →ₗ[R] N := Submodule.liftQ _ ((LinearMap.lsmul R N).flip n) <| fun r hr => by
    simpa only [LinearMap.mem_ker, LinearMap.flip_apply, LinearMap.lsmul_apply,
      Submodule.mem_annihilator_span_singleton, I] using hr
  have injective_inc : Function.Injective inc := LinearMap.ker_eq_bot.1 <| eq_bot_iff.2 <| by
    intro r hr
    induction r using Quotient.inductionOn' with | h r =>
    simpa only [Submodule.Quotient.mk''_eq_mk, Submodule.mem_bot, Submodule.Quotient.mk_eq_zero,
      Submodule.mem_annihilator_span_singleton, LinearMap.mem_ker, Submodule.liftQ_apply,
      LinearMap.flip_apply, LinearMap.lsmul_apply, I, inc] using hr
  /-
    case neg
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    fl : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Nontrivial N
    n : N
    hn : Ne n 0
    I : Ideal R := (Submodule.span R (Singleton.singleton n)).annihilator
    I_ne_top : Not (Eq I Top.top)
    inc : LinearMap (RingHom.id R) (HasQuotient.Quotient R I) N := Submodule.liftQ …
    injective_inc : Function.Injective ⇑inc
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  have ne_top := iff_flat_and_proper_ideal R M |>.1 fl |>.2 I I_ne_top
  /-
    case neg
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    fl : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Nontrivial N
    n : N
    hn : Ne n 0
    I : Ideal R := (Submodule.span R (Singleton.singleton n)).annihilator
    I_ne_top : Not (Eq I Top.top)
    inc : LinearMap (RingHom.id R) (HasQuotient.Quotient R I) N := Submodule.liftQ …
    injective_inc : Function.Injective ⇑inc
    ne_top : Ne (HSMul.hSMul I Top.top) Top.top
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  refine subsingleton_or_nontrivial _ |>.resolve_left fun rid => ?_
  exact False.elim <| ne_top <| Submodule.subsingleton_quotient_iff_eq_top.1 <|
    Function.Injective.comp (g := LinearMap.rTensor M inc)
      (fl.toFlat.rTensor_preserves_injective_linearMap inc injective_inc)
      ((quotTensorEquivQuotSMul M I).symm.injective) |>.subsingleton


instance lTensor_nontrivial
    [FaithfullyFlat R M] (N : Type*) [AddCommGroup N] [Module R N] [Nontrivial N] :
    Nontrivial (M ⊗[R] N) :=
  TensorProduct.comm R M N |>.toEquiv.nontrivial


lemma rTensor_reflects_triviality
    [FaithfullyFlat R M] (N : Type*) [AddCommGroup N] [Module R N]
    [h : Subsingleton (N ⊗[R] M)] : Subsingleton N := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    h : Subsingleton (TensorProduct R N M)
    ⊢ Subsingleton N
  -/
  revert h; change _ → _; contrapose
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ Not (Subsingleton N) → Not (Subsingleton (TensorProduct R N M))
  -/
  simp only [not_subsingleton_iff_nontrivial]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ Nontrivial N → Nontrivial (TensorProduct R N M)
  -/
  intro h
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    h : Nontrivial N
    ⊢ Nontrivial (TensorProduct R N M)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma lTensor_reflects_triviality
    [FaithfullyFlat R M] (N : Type*) [AddCommGroup N] [Module R N]
    [Subsingleton (M ⊗[R] N)] :
    Subsingleton N := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Subsingleton (TensorProduct R M N)
    ⊢ Subsingleton N
  -/
  haveI : Subsingleton (N ⊗[R] M) := (TensorProduct.comm R N M).toEquiv.injective.subsingleton
  /-
    R : Type u
    M : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.FaithfullyFlat R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Subsingleton (TensorProduct R M N)
    this : Subsingleton (TensorProduct R N M)
    ⊢ Subsingleton N
  -/
  apply rTensor_reflects_triviality R M
  /-
    🎉 no goals
  -/


attribute [-simp] Ideal.Quotient.mk_eq_mk in
lemma iff_flat_and_rTensor_faithful :
    FaithfullyFlat R M ↔
    (Flat R M ∧
      ∀ (N : Type max u v) [AddCommGroup N] [Module R N],
        Nontrivial N → Nontrivial (N ⊗[R] M)) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.FaithfullyFlat R M) (And (Module.Flat R M) (∀ (N : Type (max u v …
  -/
  refine ⟨fun fl => ⟨inferInstance, rTensor_nontrivial R M⟩, fun ⟨flat, faithful⟩ => ⟨?_⟩⟩
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ (N : Type (max u v)) [inst : AddCommGroup N] [in …
    flat : Module.Flat R M
    faithful : ∀ (N : Type (max u v)) [inst : AddCommGroup N] [inst_1 : Module R N …
    ⊢ ∀ ⦃m : Ideal R⦄, m.IsMaximal → Ne (HSMul.hSMul m Top.top) Top.top
  -/
  intro m hm rid
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ (N : Type (max u v)) [inst : AddCommGroup N] [in …
    flat : Module.Flat R M
    faithful : ∀ (N : Type (max u v)) [inst : AddCommGroup N] [inst_1 : Module R N …
    m : Ideal R
    hm : m.IsMaximal
    rid : Eq (HSMul.hSMul m Top.top) Top.top
    ⊢ False
  -/
  specialize faithful (ULift (R ⧸ m)) inferInstance
  haveI : Nontrivial ((R ⧸ m) ⊗[R] M) :=
    (congr (ULift.moduleEquiv : ULift (R ⧸ m) ≃ₗ[R] R ⧸ m)
      (LinearEquiv.refl R M)).symm.toEquiv.nontrivial
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ (N : Type (max u v)) [inst : AddCommGroup N] [in …
    flat : Module.Flat R M
    m : Ideal R
    hm : m.IsMaximal
    rid : Eq (HSMul.hSMul m Top.top) Top.top
    faithful : Nontrivial (TensorProduct R (ULift.{v, u} (HasQuotient.Quotient R m …
    this : Nontrivial (TensorProduct R (HasQuotient.Quotient R m) M)
    ⊢ False
  -/
  have := (quotTensorEquivQuotSMul M m).toEquiv.symm.nontrivial
  haveI H : Subsingleton (M ⧸ m • (⊤ : Submodule R M)) := by
    rwa [Submodule.subsingleton_quotient_iff_eq_top]
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ (N : Type (max u v)) [inst : AddCommGroup N] [in …
    flat : Module.Flat R M
    m : Ideal R
    hm : m.IsMaximal
    rid : Eq (HSMul.hSMul m Top.top) Top.top
    faithful : Nontrivial (TensorProduct R (ULift.{v, u} (HasQuotient.Quotient R m …
    this✝ : Nontrivial (TensorProduct R (HasQuotient.Quotient R m) M)
    this : Nontrivial (HasQuotient.Quotient M (HSMul.hSMul m Top.top))
    H : Subsingleton (HasQuotient.Quotient M (HSMul.hSMul m Top.top))
    ⊢ False
  -/
  rw [← not_nontrivial_iff_subsingleton] at H
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : And (Module.Flat R M) (∀ (N : Type (max u v)) [inst : AddCommGroup N] [in …
    flat : Module.Flat R M
    m : Ideal R
    hm : m.IsMaximal
    rid : Eq (HSMul.hSMul m Top.top) Top.top
    faithful : Nontrivial (TensorProduct R (ULift.{v, u} (HasQuotient.Quotient R m …
    this✝ : Nontrivial (TensorProduct R (HasQuotient.Quotient R m) M)
    this : Nontrivial (HasQuotient.Quotient M (HSMul.hSMul m Top.top))
    H : Not (Nontrivial (HasQuotient.Quotient M (HSMul.hSMul m Top.top)))
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


lemma iff_flat_and_rTensor_reflects_triviality :
    FaithfullyFlat R M ↔
    (Flat R M ∧
      ∀ (N : Type max u v) [AddCommGroup N] [Module R N],
        Subsingleton (N ⊗[R] M) → Subsingleton N) :=
  iff_flat_and_rTensor_faithful R M |>.trans <| and_congr_right_iff.2 fun _ => iff_of_eq <|
    forall_congr fun N => forall_congr fun _ => forall_congr fun _ => iff_iff_eq.1 <| by
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝² : Module.Flat R M
        N : Type (max u v)
        x✝¹ : AddCommGroup N
        x✝ : Module R N
        ⊢ Iff (Nontrivial N → Nontrivial (TensorProduct R N M)) (Subsingleton (TensorP …
      -/
      simp only [← not_subsingleton_iff_nontrivial]; tauto
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma iff_flat_and_lTensor_faithful :
    FaithfullyFlat R M ↔
    (Flat R M ∧
      ∀ (N : Type max u v) [AddCommGroup N] [Module R N],
        Nontrivial N → Nontrivial (M ⊗[R] N)) :=
  iff_flat_and_rTensor_faithful R M |>.trans
  ⟨fun ⟨flat, faithful⟩ => ⟨flat, fun N _ _ _ =>
      letI := faithful N inferInstance; (TensorProduct.comm R M N).toEquiv.nontrivial⟩,
    fun ⟨flat, faithful⟩ => ⟨flat, fun N _ _ _ =>
      letI := faithful N inferInstance; (TensorProduct.comm R M N).symm.toEquiv.nontrivial⟩⟩


lemma iff_flat_and_lTensor_reflects_triviality :
    FaithfullyFlat R M ↔
    (Flat R M ∧
      ∀ (N : Type max u v) [AddCommGroup N] [Module R N],
        Subsingleton (M ⊗[R] N) → Subsingleton N) :=
  iff_flat_and_lTensor_faithful R M |>.trans <| and_congr_right_iff.2 fun _ => iff_of_eq <|
    forall_congr fun N => forall_congr fun _ => forall_congr fun _ => iff_iff_eq.1 <| by
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝² : Module.Flat R M
        N : Type (max u v)
        x✝¹ : AddCommGroup N
        x✝ : Module R N
        ⊢ Iff (Nontrivial N → Nontrivial (TensorProduct R M N)) (Subsingleton (TensorP …
      -/
      simp only [← not_subsingleton_iff_nontrivial]; tauto
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If `M` is a faithfully flat `R`-module and `N` is `R`-linearly isomorphic to `M`, then
`N` is faithfully flat. -/
lemma of_linearEquiv {N : Type*} [AddCommGroup N] [Module R N] [FaithfullyFlat R M]
    (e : N ≃ₗ[R] M) : FaithfullyFlat R N := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : Module.FaithfullyFlat R M
    e : LinearEquiv (RingHom.id R) N M
    ⊢ Module.FaithfullyFlat R N
  -/
  rw [iff_flat_and_lTensor_faithful]
  exact ⟨Flat.of_linearEquiv R M N e,
    fun P _ _ hP ↦ (TensorProduct.congr e (LinearEquiv.refl R P)).toEquiv.nontrivial⟩


/-- A direct sum of faithfully flat `R`-modules is faithfully flat. -/
instance directSum {ι : Type*} [Nonempty ι] (M : ι → Type*) [∀ i, AddCommGroup (M i)]
    [∀ i, Module R (M i)] [∀ i, FaithfullyFlat R (M i)] : FaithfullyFlat R (⨁ i, M i) := by
  classical
  rw [iff_flat_and_lTensor_faithful]
  refine ⟨inferInstance, fun N _ _ hN ↦ ?_⟩
  obtain ⟨i⟩ := ‹Nonempty ι›
  obtain ⟨x, y, hxy⟩ := Nontrivial.exists_pair_ne (α := M i ⊗[R] N)
  haveI : Nontrivial (⨁ (i : ι), M i ⊗[R] N) :=
    ⟨DirectSum.of _ i x, DirectSum.of _ i y, fun h ↦ hxy (DirectSum.of_injective i h)⟩
  apply (TensorProduct.directSumLeft R M N).toEquiv.nontrivial


/-- Free `R`-modules over discrete types are flat. -/
instance finsupp (ι : Type v) [Nonempty ι] : FaithfullyFlat R (ι →₀ R) := by
  /-
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type v
    inst✝ : Nonempty ι
    ⊢ Module.FaithfullyFlat R (Finsupp ι R)
  -/
  classical exact of_linearEquiv _ _ (finsuppLEquivDirectSum R R ι)
  /-
    🎉 no goals
  -/


/-- Any free, nontrivial `R`-module is flat. -/
instance [Nontrivial M] [Module.Free R M] : FaithfullyFlat R M :=
  of_linearEquiv _ _ (Free.repr R M)


/--
If `M` is faithfully flat, then exactness of `N₁ ⊗ M -> N₂ ⊗ M -> N₃ ⊗ M` implies that the
composition `N₁ -> N₂ -> N₃` is `0`.

Implementation detail, please use `rTensor_reflects_exact` instead.
-/
lemma range_le_ker_of_exact_rTensor [fl : FaithfullyFlat R M]
    (ex : Function.Exact (l12.rTensor M) (l23.rTensor M)) :
    LinearMap.range l12 ≤ LinearMap.ker l23 := by
  -- let `n1 ∈ N1`. We need to show `l23 (l12 n1) = 0`. Suppose this is not the case.
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    ⊢ LE.le (LinearMap.range l12) (LinearMap.ker l23)
  -/
  rintro _ ⟨n1, rfl⟩
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    n1 : N1
    ⊢ Membership.mem (LinearMap.ker l23) (l12 n1)
  -/
  rw [LinearMap.mem_ker]
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    n1 : N1
    ⊢ Eq (l23 (l12 n1)) 0
  -/
  by_contra! hn1
  -- Let `E` be the submodule spanned by `l23 (l12 n1)`. Then because `l23 (l12 n1) ≠ 0`, we have
  -- `E ≠ 0`.
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    n1 : N1
    hn1 : Ne (l23 (l12 n1)) 0
    ⊢ False
  -/
  let E : Submodule R N3 := Submodule.span R {l23 (l12 n1)}
  have hE : Nontrivial E :=
    ⟨0, ⟨⟨l23 (l12 n1), Submodule.mem_span_singleton_self _⟩, Subtype.coe_ne_coe.1 hn1.symm⟩⟩

  -- Since `N1 ⊗ M -> N2 ⊗ M -> N3 ⊗ M` is exact, we have `l23 (l12 n1) ⊗ₜ m = 0` for all `m : M`.
  have eq1 : ∀ (m : M), l23 (l12 n1) ⊗ₜ[R] m = 0 := fun m ↦
    ex.apply_apply_eq_zero (n1 ⊗ₜ[R] m)
  -- Then `E ⊗ M = 0`. Indeed,
  have eq0 : (⊤ : Submodule R (E ⊗[R] M)) = ⊥ := by
    -- suppose `x ∈ E ⊗ M`. We will show `x = 0`.
    ext x
    simp only [Submodule.mem_top, Submodule.mem_bot, true_iff]
    have mem : x ∈ (⊤ : Submodule R _) := ⟨⟩
    rw [← TensorProduct.span_tmul_eq_top, mem_span_set] at mem
    obtain ⟨c, hc, rfl⟩ := mem
    choose b a hy using hc
    let r :  ⦃a : E ⊗[R] M⦄ → a ∈ ↑c.support → R := fun a ha =>
      Submodule.mem_span_singleton.1 (b ha).2 |>.choose
    have hr : ∀ ⦃i : E ⊗[R] M⦄ (hi : i ∈ c.support), b hi =
        r hi • ⟨l23 (l12 n1), Submodule.mem_span_singleton_self _⟩ := fun a ha =>
      Subtype.ext <| Submodule.mem_span_singleton.1 (b ha).2 |>.choose_spec.symm
    -- Since `M` is flat and `E -> N1` is injective, we only need to check that x = 0
    -- in `N1 ⊗ M`. We write `x = ∑ μᵢ • (l23 (l12 n1)) ⊗ mᵢ = ∑ μᵢ • 0 = 0`
    -- (remember `E = span {l23 (l12 n1)}` and `eq1`)
    refine Finset.sum_eq_zero fun i hi => show c i • i = 0 from
      (Module.Flat.rTensor_preserves_injective_linearMap (M := M) E.subtype <|
              Submodule.injective_subtype E) ?_
    rw [← hy hi, hr hi, smul_tmul, map_smul, LinearMap.rTensor_tmul, Submodule.subtype_apply, eq1,
      smul_zero, map_zero]
  have : Subsingleton (E ⊗[R] M) := subsingleton_iff_forall_eq 0 |>.2 fun x =>
    show x ∈ (⊥ : Submodule R _) from eq0 ▸ ⟨⟩

  -- but `E ⊗ M = 0` implies `E = 0` because `M` is faithfully flat and this is a contradiction.
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    n1 : N1
    hn1 : Ne (l23 (l12 n1)) 0
    E : Submodule R N3 := Submodule.span R (Singleton.singleton (l23 (l12 n1)))
    hE : Nontrivial (Subtype fun x => Membership.mem E x)
    eq1 : ∀ (m : M), Eq (TensorProduct.tmul R (l23 (l12 n1)) m) 0
    eq0 : Eq Top.top Bot.bot
    this : Subsingleton (TensorProduct R (Subtype fun x => Membership.mem E x) M)
    ⊢ False
  -/
  exact not_subsingleton_iff_nontrivial.2 inferInstance <| fl.rTensor_reflects_triviality R M E
  /-
    🎉 no goals
  -/


lemma rTensor_reflects_exact [fl : FaithfullyFlat R M]
    (ex : Function.Exact (l12.rTensor M) (l23.rTensor M)) :
    Function.Exact l12 l23 := LinearMap.exact_iff.2 <| by
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    ⊢ Eq (LinearMap.ker l23) (LinearMap.range l12)
  -/
  have complex : LinearMap.range l12 ≤ LinearMap.ker l23 := range_le_ker_of_exact_rTensor R M _ _ ex
  -- By the previous lemma we have that range l12 ≤ ker l23 and hence the quotient
  -- H := ker l23 ⧸ range l12 makes sense.
  -- Hence our goal ker l23 = range l12 follows from the claim that H = 0.
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    complex : LE.le (LinearMap.range l12) (LinearMap.ker l23)
    ⊢ Eq (LinearMap.ker l23) (LinearMap.range l12)
  -/
  let H := LinearMap.ker l23 ⧸ LinearMap.range (Submodule.inclusion complex)
  suffices triv_coh : Subsingleton H by
    rw [Submodule.subsingleton_quotient_iff_eq_top, Submodule.range_inclusion,
      Submodule.comap_subtype_eq_top] at triv_coh
    exact le_antisymm triv_coh complex

  -- Since `M` is faithfully flat, we need only to show that `H ⊗ M` is trivial.
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    complex : LE.le (LinearMap.range l12) (LinearMap.ker l23)
    H : Type u_2 := HasQuotient.Quotient (Subtype fun x => Membership.mem (LinearM …
    ⊢ Subsingleton H
  -/
  suffices Subsingleton (H ⊗[R] M) from rTensor_reflects_triviality R M H
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    complex : LE.le (LinearMap.range l12) (LinearMap.ker l23)
    H : Type u_2 := HasQuotient.Quotient (Subtype fun x => Membership.mem (LinearM …
    ⊢ Subsingleton (TensorProduct R H M)
  -/
  let e : H ⊗[R] M ≃ₗ[R] _ := TensorProduct.quotientTensorEquiv _ _
  -- Note that `H ⊗ M` is isomorphic to `ker l12 ⊗ M ⧸ range ((range l12 ⊗ M) -> (ker l23 ⊗ M))`.
  -- So the problem is reduced to proving surjectivity of `range l12 ⊗ M → ker l23 ⊗ M`.
  rw [e.toEquiv.subsingleton_congr, Submodule.subsingleton_quotient_iff_eq_top,
    LinearMap.range_eq_top]
  /-
    R : Type u
    M : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N1 : Type u_1
    inst✝⁵ : AddCommGroup N1
    inst✝⁴ : Module R N1
    N2 : Type u_2
    inst✝³ : AddCommGroup N2
    inst✝² : Module R N2
    N3 : Type u_3
    inst✝¹ : AddCommGroup N3
    inst✝ : Module R N3
    l12 : LinearMap (RingHom.id R) N1 N2
    l23 : LinearMap (RingHom.id R) N2 N3
    fl : Module.FaithfullyFlat R M
    ex : Function.Exact ⇑(LinearMap.rTensor M l12) ⇑(LinearMap.rTensor M l23)
    complex : LE.le (LinearMap.range l12) (LinearMap.ker l23)
    H : Type u_2 := HasQuotient.Quotient (Subtype fun x => Membership.mem (LinearM …
    e : LinearEquiv (RingHom.id R) (TensorProduct R H M) (HasQuotient.Quotient (Te …
    ⊢ Function.Surjective ⇑(TensorProduct.map (LinearMap.range (Submodule.inclusio …
  -/
  intro x
  induction x using TensorProduct.induction_on with
  | zero => exact ⟨0, by simp⟩
  -- let `x ⊗ m` be an element in `ker l23 ⊗ M`, then `x ⊗ m` is in the kernel of `l23 ⊗ 𝟙M`.
  -- Since `N1 ⊗ M -l12 ⊗ M-> N2 ⊗ M -l23 ⊗ M-> N3 ⊗ M` is exact, we have that `x ⊗ m` is in
  -- the range of `l12 ⊗ 𝟙M`, i.e. `x ⊗ m = (l12 ⊗ 𝟙M) y` for some `y ∈ N1 ⊗ M` as elements of
  -- `N2 ⊗ M`. We need to prove that `x ⊗ m = (l12 ⊗ 𝟙M) y` still holds in `(ker l23) ⊗ M`.
  -- This is okay because `M` is flat and `ker l23 -> N2` is injective.
  | tmul x m =>
    rcases x with ⟨x, (hx : l23 x = 0)⟩
    have mem : x ⊗ₜ[R] m ∈ LinearMap.ker (l23.rTensor M) := by simp [hx]
    rw [LinearMap.exact_iff.1 ex] at mem
    obtain ⟨y, hy⟩ := mem

    refine ⟨LinearMap.rTensor M (LinearMap.rangeRestrict _ ∘ₗ LinearMap.rangeRestrict l12) y,
      Module.Flat.rTensor_preserves_injective_linearMap (LinearMap.ker l23).subtype
      Subtype.val_injective ?_⟩
    simp only [LinearMap.comp_codRestrict, LinearMap.rTensor_tmul, Submodule.coe_subtype, ← hy]
    rw [← LinearMap.comp_apply]
    erw [← LinearMap.rTensor_comp]
    rw [← LinearMap.comp_apply, ← LinearMap.rTensor_comp, LinearMap.comp_assoc,
      LinearMap.subtype_comp_codRestrict, ← LinearMap.comp_assoc, Submodule.subtype_comp_inclusion,
      LinearMap.subtype_comp_codRestrict]
  | add x y hx hy =>
    obtain ⟨x, rfl⟩ := hx; obtain ⟨y, rfl⟩ := hy
    exact ⟨x + y, by simp⟩


lemma lTensor_reflects_exact [fl : FaithfullyFlat R M]
    (ex : Function.Exact (l12.lTensor M) (l23.lTensor M)) :
    Function.Exact l12 l23 :=
  rTensor_reflects_exact R M _ _ <| ex.of_ladder_linearEquiv_of_exact
    (e₁ := TensorProduct.comm _ _ _) (e₂ := TensorProduct.comm _ _ _)
                                         /-
                                           R : Type u
                                           M : Type v
                                           inst✝⁸ : CommRing R
                                           inst✝⁷ : AddCommGroup M
                                           inst✝⁶ : Module R M
                                           N1 : Type u_1
                                           inst✝⁵ : AddCommGroup N1
                                           inst✝⁴ : Module R N1
                                           N2 : Type u_2
                                           inst✝³ : AddCommGroup N2
                                           inst✝² : Module R N2
                                           N3 : Type u_3
                                           inst✝¹ : AddCommGroup N3
                                           inst✝ : Module R N3
                                           l12 : LinearMap (RingHom.id R) N1 N2
                                           l23 : LinearMap (RingHom.id R) N2 N3
                                           fl : Module.FaithfullyFlat R M
                                           ex : Function.Exact ⇑(LinearMap.lTensor M l12) ⇑(LinearMap.lTensor M l23)
                                           ⊢ Eq ((LinearMap.rTensor M l12).comp ↑(TensorProduct.comm R M N1)) ((↑(TensorP …
                                         -/
                                              /-
                                                🎉 no goals
                                              -/
    (e₃ := TensorProduct.comm _ _ _) (by ext; rfl) (by ext; rfl)
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma exact_iff_rTensor_exact [fl : FaithfullyFlat R M]
    {N1 : Type max u v} [AddCommGroup N1] [Module R N1]
    {N2 : Type max u v} [AddCommGroup N2] [Module R N2]
    {N3 : Type max u v} [AddCommGroup N3] [Module R N3]
    (l12 : N1 →ₗ[R] N2) (l23 : N2 →ₗ[R] N3) :
    Function.Exact l12 l23 ↔ Function.Exact (l12.rTensor M) (l23.rTensor M) :=
  ⟨fun e => Module.Flat.iff_rTensor_exact.1 fl.toFlat e,
    fun ex => rTensor_reflects_exact R M l12 l23 ex⟩


lemma iff_exact_iff_rTensor_exact :
    FaithfullyFlat R M ↔
    (∀ {N1 : Type max u v} [AddCommGroup N1] [Module R N1]
      {N2 : Type max u v} [AddCommGroup N2] [Module R N2]
      {N3 : Type max u v} [AddCommGroup N3] [Module R N3]
      (l12 : N1 →ₗ[R] N2) (l23 : N2 →ₗ[R] N3),
        Function.Exact l12 l23 ↔ Function.Exact (l12.rTensor M) (l23.rTensor M)) :=
  ⟨fun fl => exact_iff_rTensor_exact R M, fun iff_exact =>
                                                                                      /-
                                                                                        R : Type u
                                                                                        M : Type v
                                                                                        inst✝² : CommRing R
                                                                                        inst✝¹ : AddCommGroup M
                                                                                        inst✝ : Module R M
                                                                                        iff_exact : ∀ {N1 : Type (max u v)} [inst : AddCommGroup N1] [inst_1 : Module  …
                                                                                        ⊢ ∀ ⦃N N' N'' : Type (max u v)⦄ [inst : AddCommGroup N] [inst_1 : AddCommGroup …
                                                                                      -/
    iff_flat_and_rTensor_reflects_triviality _ _ |>.2 ⟨Flat.iff_rTensor_exact.2 <| by aesop,
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    fun N _ _ h => subsingleton_iff_forall_eq 0 |>.2 <| fun y => by
      simpa [eq_comm] using (iff_exact (0 : PUnit →ₗ[R] N) (0 : N →ₗ[R] PUnit) |>.2 fun x => by
        simpa using Subsingleton.elim _ _) y⟩⟩


lemma iff_exact_iff_lTensor_exact :
    FaithfullyFlat R M ↔
    (∀ {N1 : Type max u v} [AddCommGroup N1] [Module R N1]
      {N2 : Type max u v} [AddCommGroup N2] [Module R N2]
      {N3 : Type max u v} [AddCommGroup N3] [Module R N3]
      (l12 : N1 →ₗ[R] N2) (l23 : N2 →ₗ[R] N3),
        Function.Exact l12 l23 ↔ Function.Exact (l12.lTensor M) (l23.lTensor M)) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Module.FaithfullyFlat R M) (∀ {N1 : Type (max u v)} [inst : AddCommGrou …
  -/
  simp only [iff_exact_iff_rTensor_exact, LinearMap.rTensor_exact_iff_lTensor_exact]
  /-
    🎉 no goals
  -/


/--
If `M` is a faithfully flat module, then for all linear maps `f`, the map `id ⊗ f = 0`, if and only
if `f = 0`. -/
lemma zero_iff_lTensor_zero [h: FaithfullyFlat R M]
    {N : Type*} [AddCommGroup N] [Module R N]
    {N' : Type*} [AddCommGroup N'] [Module R N'] (f : N →ₗ[R] N') :
    f = 0 ↔ LinearMap.lTensor M f = 0 :=
  ⟨fun hf => hf.symm ▸ LinearMap.lTensor_zero M, fun hf => by
    have := lTensor_reflects_exact R M f LinearMap.id (by
      rw [LinearMap.exact_iff, hf, LinearMap.range_zero, LinearMap.ker_eq_bot]
      apply Module.Flat.lTensor_preserves_injective_linearMap
      exact fun _ _ h => h)
    /-
      R : Type u
      M : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      h : Module.FaithfullyFlat R M
      N : Type u_1
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      N' : Type u_2
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      f : LinearMap (RingHom.id R) N N'
      hf : Eq (LinearMap.lTensor M f) 0
      this : Function.Exact ⇑f ⇑LinearMap.id
      ⊢ Eq f 0
    -/
    ext x; simpa using this (f x)⟩
           /-
             🎉 no goals
           -/



/--
If `M` is a faithfully flat module, then for all linear maps `f`, the map `f ⊗ id = 0`, if and only
if `f = 0`. -/
lemma zero_iff_rTensor_zero [h: FaithfullyFlat R M]
    {N : Type*} [AddCommGroup N] [Module R N]
    {N' : Type*} [AddCommGroup N'] [Module R N']
    (f : N →ₗ[R] N') :
    f = 0 ↔ LinearMap.rTensor M f = 0 :=
  zero_iff_lTensor_zero R M f |>.trans
               /-
                 R : Type u
                 M : Type v
                 inst✝⁶ : CommRing R
                 inst✝⁵ : AddCommGroup M
                 inst✝⁴ : Module R M
                 h✝ : Module.FaithfullyFlat R M
                 N : Type u_1
                 inst✝³ : AddCommGroup N
                 inst✝² : Module R N
                 N' : Type u_2
                 inst✝¹ : AddCommGroup N'
                 inst✝ : Module R N'
                 f : LinearMap (RingHom.id R) N N'
                 h : Eq (LinearMap.lTensor M f) 0
                 ⊢ Eq (LinearMap.rTensor M f) 0
               -/
  ⟨fun h => by ext n m; exact (TensorProduct.comm R N' M).injective <|
    (by simpa using congr($h (m ⊗ₜ n))), fun h => by
    /-
      R : Type u
      M : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      h✝ : Module.FaithfullyFlat R M
      N : Type u_1
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      N' : Type u_2
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      f : LinearMap (RingHom.id R) N N'
      h : Eq (LinearMap.rTensor M f) 0
      ⊢ Eq (LinearMap.lTensor M f) 0
    -/
    ext m n; exact (TensorProduct.comm R M N').injective <| (by simpa using congr($h (n ⊗ₜ m)))⟩
             /-
               🎉 no goals
             -/


/--
An `R`-module `M` is faithfully flat iff it is flat and for all linear maps `f`, the map
`id ⊗ f = 0`, if and only if `f = 0`. -/
lemma iff_zero_iff_lTensor_zero :
    FaithfullyFlat R M ↔
    (Module.Flat R M ∧
      (∀ {N : Type max u v} [AddCommGroup N] [Module R N]
        {N' : Type max u v} [AddCommGroup N'] [Module R N']
        (f : N →ₗ[R] N'), f.lTensor M = 0 ↔ f = 0)) :=
  ⟨fun fl => ⟨inferInstance, fun f => zero_iff_lTensor_zero R M f |>.symm⟩,
    fun ⟨flat, Z⟩ => iff_flat_and_lTensor_reflects_triviality R M |>.2 ⟨flat, fun N _ _ _ => by
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R M N)
        ⊢ Subsingleton N
      -/
      have := Z (LinearMap.id : N →ₗ[R] N) |>.1 (by ext; exact Subsingleton.elim _ _)
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R M N)
        this : Eq LinearMap.id 0
        ⊢ Subsingleton N
      -/
      rw [subsingleton_iff_forall_eq 0]
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R M N)
        this : Eq LinearMap.id 0
        ⊢ ∀ (y : N), Eq y 0
      -/
      exact fun y => congr($this y)⟩⟩
      /-
        🎉 no goals
      -/


/--
An `R`-module `M` is faithfully flat iff it is flat and for all linear maps `f`, the map
`id ⊗ f = 0`, if and only if `f = 0`. -/
lemma iff_zero_iff_rTensor_zero :
    FaithfullyFlat R M ↔
    (Module.Flat R M ∧
      (∀ {N : Type max u v} [AddCommGroup N] [Module R N]
        {N' : Type max u v} [AddCommGroup N'] [Module R N']
        (f : N →ₗ[R] N'), f.rTensor M = 0 ↔ (f = 0))) :=
  ⟨fun fl => ⟨inferInstance, fun f => zero_iff_rTensor_zero R M f |>.symm⟩,
    fun ⟨flat, Z⟩ => iff_flat_and_rTensor_reflects_triviality R M |>.2 ⟨flat, fun N _ _ _ => by
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R N M)
        ⊢ Subsingleton N
      -/
      have := Z (LinearMap.id : N →ₗ[R] N) |>.1 (by ext; exact Subsingleton.elim _ _)
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R N M)
        this : Eq LinearMap.id 0
        ⊢ Subsingleton N
      -/
      rw [subsingleton_iff_forall_eq 0]
      /-
        R : Type u
        M : Type v
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        x✝³ : And (Module.Flat R M) (∀ {N : Type (max u v)} [inst : AddCommGroup N] [i …
        flat : Module.Flat R M
        Z : ∀ {N : Type (max u v)} [inst : AddCommGroup N] [inst_1 : Module R N] {N' : …
        N : Type (max u v)
        x✝² : AddCommGroup N
        x✝¹ : Module R N
        x✝ : Subsingleton (TensorProduct R N M)
        this : Eq LinearMap.id 0
        ⊢ ∀ (y : N), Eq y 0
      -/
      exact fun y => congr($this y)⟩⟩
      /-
        🎉 no goals
      -/


include S in
/-- If `S` is a faithfully flat `R`-algebra, then any faithfully flat `S`-Module is faithfully flat
as an `R`-module. -/
theorem trans : FaithfullyFlat R M := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    S : Type u_2
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.FaithfullyFlat R S
    inst✝ : Module.FaithfullyFlat S M
    ⊢ Module.FaithfullyFlat R M
  -/
  rw [iff_zero_iff_lTensor_zero]
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    S : Type u_2
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.FaithfullyFlat R S
    inst✝ : Module.FaithfullyFlat S M
    ⊢ And (Module.Flat R M) (∀ {N : Type (max u_1 u_3)} [inst : AddCommGroup N] [i …
  -/
  refine ⟨Module.Flat.trans R S M, @fun N _ _ N' _ _ f => ⟨fun aux => ?_, fun eq => eq ▸ by simp⟩⟩
  rw [zero_iff_lTensor_zero (R:= R) (M := S) f,
    show f.lTensor S = (AlgebraTensorModule.map (A:= S) LinearMap.id f).restrictScalars R by aesop,
    show (0 :  S ⊗[R] N →ₗ[R] S ⊗[R] N') = (0 : S ⊗[R] N →ₗ[S] S ⊗[R] N').restrictScalars R by rfl,
    restrictScalars_inj, zero_iff_lTensor_zero (R:= S) (M := M)]
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    S : Type u_2
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.FaithfullyFlat R S
    inst✝ : Module.FaithfullyFlat S M
    N : Type (max u_1 u_3)
    x✝³ : AddCommGroup N
    x✝² : Module R N
    N' : Type (max u_1 u_3)
    x✝¹ : AddCommGroup N'
    x✝ : Module R N'
    f : LinearMap (RingHom.id R) N N'
    aux : Eq (LinearMap.lTensor M f) 0
    ⊢ Eq (LinearMap.lTensor M (TensorProduct.AlgebraTensorModule.map LinearMap.id  …
  -/
  ext m n
  /-
    case a.h.a.h.h
    R : Type u_1
    inst✝⁸ : CommRing R
    S : Type u_2
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.FaithfullyFlat R S
    inst✝ : Module.FaithfullyFlat S M
    N : Type (max u_1 u_3)
    x✝³ : AddCommGroup N
    x✝² : Module R N
    N' : Type (max u_1 u_3)
    x✝¹ : AddCommGroup N'
    x✝ : Module R N'
    f : LinearMap (RingHom.id R) N N'
    aux : Eq (LinearMap.lTensor M f) 0
    m : M
    n : N
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry ((TensorProduct.AlgebraTensorM …
  -/
  apply_fun AlgebraTensorModule.cancelBaseChange R S S M N' using LinearEquiv.injective _
  /-
    case a.h.a.h.h
    R : Type u_1
    inst✝⁸ : CommRing R
    S : Type u_2
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    inst✝¹ : Module.FaithfullyFlat R S
    inst✝ : Module.FaithfullyFlat S M
    N : Type (max u_1 u_3)
    x✝³ : AddCommGroup N
    x✝² : Module R N
    N' : Type (max u_1 u_3)
    x✝¹ : AddCommGroup N'
    x✝ : Module R N'
    f : LinearMap (RingHom.id R) N N'
    aux : Eq (LinearMap.lTensor M f) 0
    m : M
    n : N
    ⊢ Eq ((TensorProduct.AlgebraTensorModule.cancelBaseChange R S S M N') (((Tenso …
  -/
  simpa using congr($aux (m ⊗ₜ[R] n))
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-08")] alias comp := trans


