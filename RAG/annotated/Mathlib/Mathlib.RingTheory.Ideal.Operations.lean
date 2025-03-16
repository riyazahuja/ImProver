lemma coe_span_smul {R' M' : Type*} [CommSemiring R'] [AddCommMonoid M'] [Module R' M']
    (s : Set R') (N : Submodule R' M') :
    (Ideal.span s : Set R') • N = s • N :=
  set_smul_eq_of_le _ _ _
        /-
          R' : Type u_1
          M' : Type u_2
          inst✝² : CommSemiring R'
          inst✝¹ : AddCommMonoid M'
          inst✝ : Module R' M'
          s : Set R'
          N : Submodule R' M'
          ⊢ ∀ ⦃r : R'⦄ ⦃n : M'⦄, Membership.mem (↑(Ideal.span s)) r → Membership.mem N n …
        -/
    (by rintro r n hr hn
        induction hr using Submodule.span_induction with
        | mem _ h => exact mem_set_smul_of_mem_mem h hn
        | zero => rw [zero_smul]; exact Submodule.zero_mem _
        | add _ _ _ _ ihr ihs => rw [add_smul]; exact Submodule.add_mem _ ihr ihs
        | smul _ _ hr =>
          rw [mem_span_set] at hr
          obtain ⟨c, hc, rfl⟩ := hr
          rw [Finsupp.sum, Finset.smul_sum, Finset.sum_smul]
          refine Submodule.sum_mem _ fun i hi => ?_
          rw [← mul_smul, smul_eq_mul, mul_comm, mul_smul]
          exact mem_set_smul_of_mem_mem (hc hi) <| Submodule.smul_mem _ _ hn) <|
    set_smul_mono_left _ Submodule.subset_span


lemma span_singleton_toAddSubgroup_eq_zmultiples (a : ℤ) :
    (span ℤ {a}).toAddSubgroup = AddSubgroup.zmultiples a := by
  /-
    a : Int
    ⊢ Eq (Submodule.span Int (Singleton.singleton a)).toAddSubgroup (AddSubgroup.z …
  -/
  ext i
  /-
    case h
    a i : Int
    ⊢ Iff (Membership.mem (Submodule.span Int (Singleton.singleton a)).toAddSubgro …
  -/
  simp [Ideal.mem_span_singleton', AddSubgroup.mem_zmultiples_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma _root_.Ideal.span_singleton_toAddSubgroup_eq_zmultiples (a : ℤ) :
   (Ideal.span {a}).toAddSubgroup = AddSubgroup.zmultiples a :=
  Submodule.span_singleton_toAddSubgroup_eq_zmultiples _


/-- This duplicates the global `smul_eq_mul`, but doesn't have to unfold anywhere near as much to
apply. -/
protected theorem _root_.Ideal.smul_eq_mul (I J : Ideal R) : I • J = I * J :=
  rfl


theorem smul_le_right : I • N ≤ N :=
  smul_le.2 fun r _ _ ↦ N.smul_mem r


theorem map_le_smul_top (I : Ideal R) (f : R →ₗ[R] M) :
    Submodule.map f I ≤ I • (⊤ : Submodule R M) := by
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    f : LinearMap (RingHom.id R) R M
    ⊢ LE.le (Submodule.map f I) (HSMul.hSMul I Top.top)
  -/
  rintro _ ⟨y, hy, rfl⟩
  /-
    case intro.intro
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    f : LinearMap (RingHom.id R) R M
    y : R
    hy : Membership.mem (↑I) y
    ⊢ Membership.mem (HSMul.hSMul I Top.top) (f y)
  -/
  rw [← mul_one y, ← smul_eq_mul, f.map_smul]
  /-
    case intro.intro
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    f : LinearMap (RingHom.id R) R M
    y : R
    hy : Membership.mem (↑I) y
    ⊢ Membership.mem (HSMul.hSMul I Top.top) (HSMul.hSMul y (f 1))
  -/
  exact smul_mem_smul hy mem_top
  /-
    🎉 no goals
  -/


@[simp]
theorem top_smul : (⊤ : Ideal R) • N = N :=
  le_antisymm smul_le_right fun r hri => one_smul R r ▸ smul_mem_smul mem_top hri


theorem mem_of_span_top_of_smul_mem (M' : Submodule R M) (s : Set R) (hs : Ideal.span s = ⊤) (x : M)
    (H : ∀ r : s, (r : R) • x ∈ M') : x ∈ M' := by
  suffices LinearMap.range (LinearMap.toSpanSingleton R M x) ≤ M' by
    rw [← LinearMap.toSpanSingleton_one R M x]
    exact this (LinearMap.mem_range_self _ 1)
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    H : ∀ (r : ↑s), Membership.mem M' (HSMul.hSMul (↑r) x)
    ⊢ LE.le (LinearMap.range (LinearMap.toSpanSingleton R M x)) M'
  -/
  rw [LinearMap.range_eq_map, ← hs, map_le_iff_le_comap, Ideal.span, span_le]
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    H : ∀ (r : ↑s), Membership.mem M' (HSMul.hSMul (↑r) x)
    ⊢ HasSubset.Subset s ↑(Submodule.comap (LinearMap.toSpanSingleton R M x) M')
  -/
  exact fun r hr ↦ H ⟨r, hr⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem map_smul'' (f : M →ₗ[R] M') : (I • N).map f = I • N.map f :=
  le_antisymm
    (map_le_iff_le_comap.2 <|
      smul_le.2 fun r hr n hn =>
        show f (r • n) ∈ I • N.map f from
          (f.map_smul r n).symm ▸ smul_mem_smul hr (mem_map_of_mem hn)) <|
    smul_le.2 fun r hr _ hn =>
      let ⟨p, hp, hfp⟩ := mem_map.1 hn
      hfp ▸ f.map_smul r p ▸ mem_map_of_mem (smul_mem_smul hr hp)


theorem mem_smul_top_iff (N : Submodule R M) (x : N) :
    x ∈ I • (⊤ : Submodule R N) ↔ (x : M) ∈ I • N := by
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    x : Subtype fun x => Membership.mem N x
    ⊢ Iff (Membership.mem (HSMul.hSMul I Top.top) x) (Membership.mem (HSMul.hSMul  …
  -/
  change _ ↔ N.subtype x ∈ I • N
  have : Submodule.map N.subtype (I • ⊤) = I • N := by
    rw [Submodule.map_smul'', Submodule.map_top, Submodule.range_subtype]
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    x : Subtype fun x => Membership.mem N x
    this : Eq (Submodule.map N.subtype (HSMul.hSMul I Top.top)) (HSMul.hSMul I N)
    ⊢ Iff (Membership.mem (HSMul.hSMul I Top.top) x) (Membership.mem (HSMul.hSMul  …
  -/
  rw [← this]
  /-
    R : Type u
    M : Type v
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    x : Subtype fun x => Membership.mem N x
    this : Eq (Submodule.map N.subtype (HSMul.hSMul I Top.top)) (HSMul.hSMul I N)
    ⊢ Iff (Membership.mem (HSMul.hSMul I Top.top) x) (Membership.mem (Submodule.ma …
  -/
  exact (Function.Injective.mem_set_image N.injective_subtype).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_comap_le_comap_smul (f : M →ₗ[R] M') (S : Submodule R M') (I : Ideal R) :
    I • S.comap f ≤ (I • S).comap f := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type w
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    S : Submodule R M'
    I : Ideal R
    ⊢ LE.le (HSMul.hSMul I (Submodule.comap f S)) (Submodule.comap f (HSMul.hSMul  …
  -/
  refine Submodule.smul_le.mpr fun r hr x hx => ?_
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type w
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    S : Submodule R M'
    I : Ideal R
    r : R
    hr : Membership.mem I r
    x : M
    hx : Membership.mem (Submodule.comap f S) x
    ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul I S)) (HSMul.hSMul r x)
  -/
  rw [Submodule.mem_comap] at hx ⊢
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type w
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    S : Submodule R M'
    I : Ideal R
    r : R
    hr : Membership.mem I r
    x : M
    hx : Membership.mem S (f x)
    ⊢ Membership.mem (HSMul.hSMul I S) (f (HSMul.hSMul r x))
  -/
  rw [f.map_smul]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    M' : Type w
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    S : Submodule R M'
    I : Ideal R
    r : R
    hr : Membership.mem I r
    x : M
    hx : Membership.mem S (f x)
    ⊢ Membership.mem (HSMul.hSMul I S) (HSMul.hSMul r (f x))
  -/
  exact Submodule.smul_mem_smul hr hx
  /-
    🎉 no goals
  -/


theorem mem_smul_span_singleton {I : Ideal R} {m : M} {x : M} :
    x ∈ I • span R ({m} : Set M) ↔ ∃ y ∈ I, y • m = x :=
  ⟨fun hx =>
    smul_induction_on hx
      (fun r hri _ hnm =>
        let ⟨s, hs⟩ := mem_span_singleton.1 hnm
        ⟨r * s, I.mul_mem_right _ hri, hs ▸ mul_smul r s m⟩)
      fun m1 m2 ⟨y1, hyi1, hy1⟩ ⟨y2, hyi2, hy2⟩ =>
                                        /-
                                          R : Type u
                                          M : Type v
                                          inst✝² : CommSemiring R
                                          inst✝¹ : AddCommMonoid M
                                          inst✝ : Module R M
                                          I : Ideal R
                                          m x : M
                                          hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Singleton.singleton m))) x
                                          m1 m2 : M
                                          x✝¹ : Exists fun y => And (Membership.mem I y) (Eq (HSMul.hSMul y m) m1)
                                          x✝ : Exists fun y => And (Membership.mem I y) (Eq (HSMul.hSMul y m) m2)
                                          y1 : R
                                          hyi1 : Membership.mem I y1
                                          hy1 : Eq (HSMul.hSMul y1 m) m1
                                          y2 : R
                                          hyi2 : Membership.mem I y2
                                          hy2 : Eq (HSMul.hSMul y2 m) m2
                                          ⊢ Eq (HSMul.hSMul (HAdd.hAdd y1 y2) m) (HAdd.hAdd m1 m2)
                                        -/
      ⟨y1 + y2, I.add_mem hyi1 hyi2, by rw [add_smul, hy1, hy2]⟩,
                                        /-
                                          🎉 no goals
                                        -/
    fun ⟨_, hyi, hy⟩ => hy ▸ smul_mem_smul hyi (subset_span <| Set.mem_singleton m)⟩


theorem smul_eq_map₂ : I • N = Submodule.map₂ (LinearMap.lsmul R M) I N :=
  le_antisymm (smul_le.mpr fun _m hm _n ↦ Submodule.apply_mem_map₂ _ hm)
    (map₂_le.mpr fun _m hm _n ↦ smul_mem_smul hm)


theorem span_smul_span : Ideal.span S • span R T = span R (⋃ (s ∈ S) (t ∈ T), {s • t}) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Set R
    T : Set M
    ⊢ Eq (HSMul.hSMul (Ideal.span S) (Submodule.span R T)) (Submodule.span R (Set. …
  -/
  rw [smul_eq_map₂]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Set R
    T : Set M
    ⊢ Eq (Submodule.map₂ (LinearMap.lsmul R M) (Ideal.span S) (Submodule.span R T) …
  -/
  exact (map₂_span_span _ _ _ _).trans <| congr_arg _ <| Set.image2_eq_iUnion _ _ _
  /-
    🎉 no goals
  -/


theorem ideal_span_singleton_smul (r : R) (N : Submodule R M) :
    (Ideal.span {r} : Ideal R) • N = r • N := by
  have : span R (⋃ (t : M) (_ : t ∈ N), {r • t}) = r • N := by
    convert span_eq (r • N)
    exact (Set.image_eq_iUnion _ (N : Set M)).symm
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    N : Submodule R M
    this : Eq (Submodule.span R (Set.iUnion fun t => Set.iUnion fun x => Singleton …
    ⊢ Eq (HSMul.hSMul (Ideal.span (Singleton.singleton r)) N) (HSMul.hSMul r N)
  -/
  conv_lhs => rw [← span_eq N, span_smul_span]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    r : R
    N : Submodule R M
    this : Eq (Submodule.span R (Set.iUnion fun t => Set.iUnion fun x => Singleton …
    ⊢ Eq (Submodule.span R (Set.iUnion fun s => Set.iUnion fun h => Set.iUnion fun …
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- Given `s`, a generating set of `R`, to check that an `x : M` falls in a
submodule `M'` of `x`, we only need to show that `r ^ n • x ∈ M'` for some `n` for each `r : s`. -/
theorem mem_of_span_eq_top_of_smul_pow_mem (M' : Submodule R M) (s : Set R) (hs : Ideal.span s = ⊤)
    (x : M) (H : ∀ r : s, ∃ n : ℕ, ((r : R) ^ n : R) • x ∈ M') : x ∈ M' := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    H : ∀ (r : ↑s), Exists fun n => Membership.mem M' (HSMul.hSMul (HPow.hPow (↑r) …
    ⊢ Membership.mem M' x
  -/
  choose f hf using H
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    f : ↑s → Nat
    hf : ∀ (r : ↑s), Membership.mem M' (HSMul.hSMul (HPow.hPow (↑r) (f r)) x)
    ⊢ Membership.mem M' x
  -/
  apply M'.mem_of_span_top_of_smul_mem _ (Ideal.span_range_pow_eq_top s hs f)
  /-
    case H
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    f : ↑s → Nat
    hf : ∀ (r : ↑s), Membership.mem M' (HSMul.hSMul (HPow.hPow (↑r) (f r)) x)
    ⊢ ∀ (r : ↑(Set.range fun x => HPow.hPow (↑x) (f x))), Membership.mem M' (HSMul …
  -/
  rintro ⟨_, r, hr, rfl⟩
  /-
    case H.mk.intro.refl
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    x : M
    f : ↑s → Nat
    hf : ∀ (r : ↑s), Membership.mem M' (HSMul.hSMul (HPow.hPow (↑r) (f r)) x)
    r : ↑s
    ⊢ Membership.mem M' (HSMul.hSMul (↑⟨(fun x => HPow.hPow (↑x) (f x)) r, ⋯⟩) x)
  -/
  exact hf r
  /-
    🎉 no goals
  -/


open Pointwise in
@[simp]
theorem map_pointwise_smul (r : R) (N : Submodule R M) (f : M →ₗ[R] M') :
    (r • N).map f = r • N.map f := by
  /-
    R : Type u
    M : Type v
    M' : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    r : R
    N : Submodule R M
    f : LinearMap (RingHom.id R) M M'
    ⊢ Eq (Submodule.map f (HSMul.hSMul r N)) (HSMul.hSMul r (Submodule.map f N))
  -/
  simp_rw [← ideal_span_singleton_smul, map_smul'']
  /-
    🎉 no goals
  -/


theorem mem_smul_span {s : Set M} {x : M} :
    x ∈ I • Submodule.span R s ↔ x ∈ Submodule.span R (⋃ (a ∈ I) (b ∈ s), ({a • b} : Set M)) := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    s : Set M
    x : M
    ⊢ Iff (Membership.mem (HSMul.hSMul I (Submodule.span R s)) x) (Membership.mem  …
  -/
  rw [← I.span_eq, Submodule.span_smul_span, I.span_eq]
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    s : Set M
    x : M
    ⊢ Iff (Membership.mem (Submodule.span R (Set.iUnion fun s_1 => Set.iUnion fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `x` is an `I`-multiple of the submodule spanned by `f '' s`,
then we can write `x` as an `I`-linear combination of the elements of `f '' s`. -/
theorem mem_ideal_smul_span_iff_exists_sum {ι : Type*} (f : ι → M) (x : M) :
    x ∈ I • span R (Set.range f) ↔
      ∃ (a : ι →₀ R) (_ : ∀ i, a i ∈ I), (a.sum fun i c => c • f i) = x := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    ι : Type u_4
    f : ι → M
    x : M
    ⊢ Iff (Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x) (Exi …
  -/
  constructor; swap
    /-
      case mpr
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      ⊢ (Exists fun a => Exists fun x_1 => Eq (a.sum fun i c => HSMul.hSMul c (f i)) …
    -/
  · rintro ⟨a, ha, rfl⟩
    /-
      case mpr.intro.intro
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      a : Finsupp ι R
      ha : ∀ (i : ι), Membership.mem I (a i)
      ⊢ Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) (a.sum fun i …
    -/
    exact Submodule.sum_mem _ fun c _ => smul_mem_smul (ha c) <| subset_span <| Set.mem_range_self _
    /-
      🎉 no goals
    -/
  /-
    case mp
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    ι : Type u_4
    f : ι → M
    x : M
    ⊢ Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x → Exists f …
  -/
  refine fun hx => span_induction ?_ ?_ ?_ ?_ (mem_smul_span.mp hx)
    /-
      case mp.refine_1
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ⊢ ∀ (x : M), Membership.mem (Set.iUnion fun a => Set.iUnion fun h => Set.iUnio …
    -/
  · simp only [Set.mem_iUnion, Set.mem_range, Set.mem_singleton_iff]
    /-
      case mp.refine_1
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ⊢ ∀ (x : M), (Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h = …
    -/
    rintro x ⟨y, hy, x, ⟨i, rfl⟩, rfl⟩
    /-
      case mp.refine_1.intro.intro.intro.intro.intro
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      y : R
      hy : Membership.mem I y
      i : ι
      ⊢ Exists fun a => Exists fun x => Eq (a.sum fun i c => HSMul.hSMul c (f i)) (H …
    -/
    refine ⟨Finsupp.single i y, fun j => ?_, ?_⟩
      /-
        case mp.refine_1.intro.intro.intro.intro.intro.refine_1
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        I : Ideal R
        ι : Type u_4
        f : ι → M
        x : M
        hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
        y : R
        hy : Membership.mem I y
        i j : ι
        ⊢ Membership.mem I ((Finsupp.single i y) j)
      -/
    · letI := Classical.decEq ι
      /-
        case mp.refine_1.intro.intro.intro.intro.intro.refine_1
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        I : Ideal R
        ι : Type u_4
        f : ι → M
        x : M
        hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
        y : R
        hy : Membership.mem I y
        i j : ι
        this : DecidableEq ι := Classical.decEq ι
        ⊢ Membership.mem I ((Finsupp.single i y) j)
      -/
      rw [Finsupp.single_apply]
      /-
        case mp.refine_1.intro.intro.intro.intro.intro.refine_1
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        I : Ideal R
        ι : Type u_4
        f : ι → M
        x : M
        hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
        y : R
        hy : Membership.mem I y
        i j : ι
        this : DecidableEq ι := Classical.decEq ι
        ⊢ Membership.mem I (ite (Eq i j) y 0)
      -/
      split_ifs
        /-
          case pos
          R : Type u
          M : Type v
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          I : Ideal R
          ι : Type u_4
          f : ι → M
          x : M
          hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
          y : R
          hy : Membership.mem I y
          i j : ι
          this : DecidableEq ι := Classical.decEq ι
          h✝ : Eq i j
          ⊢ Membership.mem I y
        -/
      · assumption
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u
          M : Type v
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          I : Ideal R
          ι : Type u_4
          f : ι → M
          x : M
          hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
          y : R
          hy : Membership.mem I y
          i j : ι
          this : DecidableEq ι := Classical.decEq ι
          h✝ : Not (Eq i j)
          ⊢ Membership.mem I 0
        -/
      · exact I.zero_mem
        /-
          🎉 no goals
        -/
    /-
      case mp.refine_1.intro.intro.intro.intro.intro.refine_2
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      y : R
      hy : Membership.mem I y
      i : ι
      ⊢ Eq ((Finsupp.single i y).sum fun i c => HSMul.hSMul c (f i)) (HSMul.hSMul y  …
    -/
    refine @Finsupp.sum_single_index ι R M _ _ i _ (fun i y => y • f i) ?_
    /-
      case mp.refine_1.intro.intro.intro.intro.intro.refine_2
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      y : R
      hy : Membership.mem I y
      i : ι
      ⊢ Eq ((fun i y => HSMul.hSMul y (f i)) i 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mp.refine_2
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ⊢ Exists fun a => Exists fun x => Eq (a.sum fun i c => HSMul.hSMul c (f i)) 0
    -/
  · exact ⟨0, fun _ => I.zero_mem, Finsupp.sum_zero_index⟩
    /-
      🎉 no goals
    -/
    /-
      case mp.refine_3
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ⊢ ∀ (x y : M), Membership.mem (Submodule.span R (Set.iUnion fun a => Set.iUnio …
    -/
  · rintro x y - - ⟨ax, hax, rfl⟩ ⟨ay, hay, rfl⟩
    /-
      case mp.refine_3.intro.intro.intro.intro
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ax : Finsupp ι R
      hax : ∀ (i : ι), Membership.mem I (ax i)
      ay : Finsupp ι R
      hay : ∀ (i : ι), Membership.mem I (ay i)
      ⊢ Exists fun a => Exists fun x => Eq (a.sum fun i c => HSMul.hSMul c (f i)) (H …
    -/
    refine ⟨ax + ay, fun i => I.add_mem (hax i) (hay i), Finsupp.sum_add_index' ?_ ?_⟩ <;>
      /-
        case mp.refine_3.intro.intro.intro.intro.refine_1
        R : Type u
        M : Type v
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        I : Ideal R
        ι : Type u_4
        f : ι → M
        x : M
        hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
        ax : Finsupp ι R
        hax : ∀ (i : ι), Membership.mem I (ax i)
        ay : Finsupp ι R
        hay : ∀ (i : ι), Membership.mem I (ay i)
        ⊢ ∀ (a : ι), Eq (HSMul.hSMul 0 (f a)) 0
      -/
                 /-
                   🎉 no goals
                 -/
      intros <;> simp only [zero_smul, add_smul]
                 /-
                   🎉 no goals
                 -/
    /-
      case mp.refine_4
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      ⊢ ∀ (a : R) (x : M), Membership.mem (Submodule.span R (Set.iUnion fun a => Set …
    -/
  · rintro c x - ⟨a, ha, rfl⟩
    /-
      case mp.refine_4.intro.intro
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      c : R
      a : Finsupp ι R
      ha : ∀ (i : ι), Membership.mem I (a i)
      ⊢ Exists fun a_1 => Exists fun x => Eq (a_1.sum fun i c => HSMul.hSMul c (f i) …
    -/
    refine ⟨c • a, fun i => I.mul_mem_left c (ha i), ?_⟩
    /-
      case mp.refine_4.intro.intro
      R : Type u
      M : Type v
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      I : Ideal R
      ι : Type u_4
      f : ι → M
      x : M
      hx : Membership.mem (HSMul.hSMul I (Submodule.span R (Set.range f))) x
      c : R
      a : Finsupp ι R
      ha : ∀ (i : ι), Membership.mem I (a i)
      ⊢ Eq ((HSMul.hSMul c a).sum fun i c => HSMul.hSMul c (f i)) (HSMul.hSMul c (a. …
    -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    rw [Finsupp.sum_smul_index, Finsupp.smul_sum] <;> intros <;> simp only [zero_smul, mul_smul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem mem_ideal_smul_span_iff_exists_sum' {ι : Type*} (s : Set ι) (f : ι → M) (x : M) :
    x ∈ I • span R (f '' s) ↔
    ∃ (a : s →₀ R) (_ : ∀ i, a i ∈ I), (a.sum fun i c => c • f i) = x := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    I : Ideal R
    ι : Type u_4
    s : Set ι
    f : ι → M
    x : M
    ⊢ Iff (Membership.mem (HSMul.hSMul I (Submodule.span R (Set.image f s))) x) (E …
  -/
  rw [← Submodule.mem_ideal_smul_span_iff_exists_sum, ← Set.image_eq_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_eq_sup {I J : Ideal R} : I + J = I ⊔ J :=
  rfl


@[simp]
theorem zero_eq_bot : (0 : Ideal R) = ⊥ :=
  rfl


@[simp]
theorem sum_eq_sup {ι : Type*} (s : Finset ι) (f : ι → Ideal R) : s.sum f = s.sup f :=
  rfl


@[simp]
theorem one_eq_top : (1 : Ideal R) = ⊤ := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq 1 Top.top
  -/
  rw [Submodule.one_eq_span, ← Ideal.span, Ideal.span_singleton_one]
  /-
    🎉 no goals
  -/


theorem add_eq_one_iff : I + J = 1 ↔ ∃ i ∈ I, ∃ j ∈ J, i + j = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    I J : Ideal R
    ⊢ Iff (Eq (HAdd.hAdd I J) 1) (Exists fun i => And (Membership.mem I i) (Exists …
  -/
  rw [one_eq_top, eq_top_iff_one, add_eq_sup, Submodule.mem_sup]
  /-
    🎉 no goals
  -/


theorem mul_mem_mul {r s} (hr : r ∈ I) (hs : s ∈ J) : r * s ∈ I * J :=
  Submodule.smul_mem_smul hr hs


theorem pow_mem_pow {x : R} (hx : x ∈ I) (n : ℕ) : x ^ n ∈ I ^ n :=
  Submodule.pow_mem_pow _ hx _


theorem mul_le : I * J ≤ K ↔ ∀ r ∈ I, ∀ s ∈ J, r * s ∈ K :=
  Submodule.smul_le


theorem mul_le_left : I * J ≤ J :=
  Ideal.mul_le.2 fun _ _ _ => J.mul_mem_left _


@[simp]
theorem sup_mul_left_self : I ⊔ J * I = I :=
  sup_eq_left.2 Ideal.mul_le_left


@[simp]
theorem mul_left_self_sup : J * I ⊔ I = I :=
  sup_eq_right.2 Ideal.mul_le_left


protected theorem mul_assoc : I * J * K = I * (J * K) :=
  Submodule.smul_assoc I J K


                                  /-
                                    R : Type u
                                    inst✝ : Semiring R
                                    I : Ideal R
                                    ⊢ Eq (HMul.hMul I Bot.bot) Bot.bot
                                  -/
theorem mul_bot : I * ⊥ = ⊥ := by simp
                                  /-
                                    🎉 no goals
                                  -/


                                  /-
                                    R : Type u
                                    inst✝ : Semiring R
                                    I : Ideal R
                                    ⊢ Eq (HMul.hMul Bot.bot I) Bot.bot
                                  -/
theorem bot_mul : ⊥ * I = ⊥ := by simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem top_mul : ⊤ * I = I :=
  Submodule.top_smul I


theorem mul_mono (hik : I ≤ K) (hjl : J ≤ L) : I * J ≤ K * L :=
  Submodule.smul_mono hik hjl


theorem mul_mono_left (h : I ≤ J) : I * K ≤ J * K :=
  Submodule.smul_mono_left h


theorem mul_mono_right (h : J ≤ K) : I * J ≤ I * K :=
  smul_mono_right I h


theorem mul_sup : I * (J ⊔ K) = I * J ⊔ I * K :=
  Submodule.smul_sup I J K


theorem sup_mul : (I ⊔ J) * K = I * K ⊔ J * K :=
  Submodule.sup_smul I J K


theorem pow_le_pow_right {m n : ℕ} (h : m ≤ n) : I ^ n ≤ I ^ m := by
  /-
    R : Type u
    inst✝ : Semiring R
    I : Ideal R
    m n : Nat
    h : LE.le m n
    ⊢ LE.le (HPow.hPow I n) (HPow.hPow I m)
  -/
  obtain _ | m := m
    /-
      case zero
      R : Type u
      inst✝ : Semiring R
      I : Ideal R
      n : Nat
      h : LE.le 0 n
      ⊢ LE.le (HPow.hPow I n) (HPow.hPow I 0)
    -/
  · rw [Submodule.pow_zero, one_eq_top]; exact le_top
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case succ
    R : Type u
    inst✝ : Semiring R
    I : Ideal R
    n m : Nat
    h : LE.le (HAdd.hAdd m 1) n
    ⊢ LE.le (HPow.hPow I n) (HPow.hPow I (HAdd.hAdd m 1))
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le h
  /-
    case succ.intro
    R : Type u
    inst✝ : Semiring R
    I : Ideal R
    m n : Nat
    h : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HAdd.hAdd m 1) n)
    ⊢ LE.le (HPow.hPow I (HAdd.hAdd (HAdd.hAdd m 1) n)) (HPow.hPow I (HAdd.hAdd m  …
  -/
  simp_rw [add_comm, (· ^ ·), Pow.pow, npowRec_add _ _ m.succ_ne_zero _ I.one_mul]
  /-
    case succ.intro
    R : Type u
    inst✝ : Semiring R
    I : Ideal R
    m n : Nat
    h : LE.le (HAdd.hAdd m 1) (HAdd.hAdd (HAdd.hAdd m 1) n)
    ⊢ LE.le (HMul.hMul (npowRec n I) (npowRec m.succ I)) (npowRec (HAdd.hAdd m 1) I)
  -/
  exact mul_le_left
  /-
    🎉 no goals
  -/


theorem pow_le_self {n : ℕ} (hn : n ≠ 0) : I ^ n ≤ I :=
  calc
    I ^ n ≤ I ^ 1 := pow_le_pow_right (Nat.pos_of_ne_zero hn)
    _ = I := Submodule.pow_one _


theorem pow_right_mono (e : I ≤ J) (n : ℕ) : I ^ n ≤ J ^ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    I J : Ideal R
    e : LE.le I J
    n : Nat
    ⊢ LE.le (HPow.hPow I n) (HPow.hPow J n)
  -/
  induction' n with _ hn
    /-
      case zero
      R : Type u
      inst✝ : Semiring R
      I J : Ideal R
      e : LE.le I J
      ⊢ LE.le (HPow.hPow I 0) (HPow.hPow J 0)
    -/
  · rw [Submodule.pow_zero, Submodule.pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      I J : Ideal R
      e : LE.le I J
      n✝ : Nat
      hn : LE.le (HPow.hPow I n✝) (HPow.hPow J n✝)
      ⊢ LE.le (HPow.hPow I (HAdd.hAdd n✝ 1)) (HPow.hPow J (HAdd.hAdd n✝ 1))
    -/
  · rw [Submodule.pow_succ, Submodule.pow_succ]
    /-
      case succ
      R : Type u
      inst✝ : Semiring R
      I J : Ideal R
      e : LE.le I J
      n✝ : Nat
      hn : LE.le (HPow.hPow I n✝) (HPow.hPow J n✝)
      ⊢ LE.le (HMul.hMul (HPow.hPow I n✝) I) (HMul.hMul (HPow.hPow J n✝) J)
    -/
    exact Ideal.mul_mono hn e
    /-
      🎉 no goals
    -/


@[simp]
theorem mul_eq_bot [NoZeroDivisors R] : I * J = ⊥ ↔ I = ⊥ ∨ J = ⊥ :=
  ⟨fun hij =>
    or_iff_not_imp_left.mpr fun I_ne_bot =>
      J.eq_bot_iff.mpr fun j hj =>
        let ⟨i, hi, ne0⟩ := I.ne_bot_iff.mp I_ne_bot
        Or.resolve_left (mul_eq_zero.mp ((I * J).eq_bot_iff.mp hij _ (mul_mem_mul hi hj))) ne0,
                /-
                  R : Type u
                  inst✝¹ : Semiring R
                  I J : Ideal R
                  inst✝ : NoZeroDivisors R
                  h : Or (Eq I Bot.bot) (Eq J Bot.bot)
                  ⊢ Eq (HMul.hMul I J) Bot.bot
                -/
    fun h => by obtain rfl | rfl := h; exacts [bot_mul _, mul_bot _]⟩
                                       /-
                                         🎉 no goals
                                       -/


instance [NoZeroDivisors R] : NoZeroDivisors (Ideal R) where
  eq_zero_or_eq_zero_of_mul_eq_zero := mul_eq_bot.1


instance {S A : Type*} [Semiring S] [SMul R S] [AddCommMonoid A] [Module R A] [Module S A]
    [IsScalarTower R S A] [NoZeroSMulDivisors R A] {I : Submodule S A} : NoZeroSMulDivisors R I :=
  Submodule.noZeroSMulDivisors (Submodule.restrictScalars R I)


theorem mul_mem_mul_rev {r s} (hr : r ∈ I) (hs : s ∈ J) : s * r ∈ I * J :=
  mul_comm r s ▸ mul_mem_mul hr hs


theorem prod_mem_prod {ι : Type*} {s : Finset ι} {I : ι → Ideal R} {x : ι → R} :
    (∀ i ∈ s, x i ∈ I i) → (∏ i ∈ s, x i) ∈ ∏ i ∈ s, I i := by
  classical
    refine Finset.induction_on s ?_ ?_
    · intro
      rw [Finset.prod_empty, Finset.prod_empty, one_eq_top]
      exact Submodule.mem_top
    · intro a s ha IH h
      rw [Finset.prod_insert ha, Finset.prod_insert ha]
      exact
        mul_mem_mul (h a <| Finset.mem_insert_self a s)
          (IH fun i hi => h i <| Finset.mem_insert_of_mem hi)


theorem mul_le_right : I * J ≤ I :=
  Ideal.mul_le.2 fun _ hr _ _ => I.mul_mem_right _ hr


@[simp, nolint simpNF]
theorem sup_mul_right_self : I ⊔ I * J = I :=
  sup_eq_left.2 Ideal.mul_le_right


@[simp, nolint simpNF]
theorem mul_right_self_sup : I * J ⊔ I = I :=
  sup_eq_right.2 Ideal.mul_le_right


lemma sup_pow_add_le_pow_sup_pow {n m : ℕ} : (I ⊔ J) ^ (n + m) ≤ I ^ n ⊔ J ^ m := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n m : Nat
    ⊢ LE.le (HPow.hPow (Max.max I J) (HAdd.hAdd n m)) (Max.max (HPow.hPow I n) (HP …
  -/
  rw [← Ideal.add_eq_sup, ← Ideal.add_eq_sup, add_pow, Ideal.sum_eq_sup]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n m : Nat
    ⊢ LE.le ((Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)).sup fun m_1 => HMul.hMul …
  -/
  apply Finset.sup_le
  /-
    case a
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n m : Nat
    ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) b → …
  -/
  intros i hi
  /-
    case a
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n m i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow I i) (HPow.hPow J (HSub.hSub (HAdd.hA …
  -/
  by_cases hn : n ≤ i
  · exact (Ideal.mul_le_right.trans (Ideal.mul_le_right.trans
      ((Ideal.pow_le_pow_right hn).trans le_sup_left)))
  · refine (Ideal.mul_le_right.trans (Ideal.mul_le_left.trans
      ((Ideal.pow_le_pow_right ?_).trans le_sup_right)))
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      n m i : Nat
      hi : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd n m) 1)) i
      hn : Not (LE.le n i)
      ⊢ LE.le m (HSub.hSub (HAdd.hAdd n m) i)
    -/
    omega
    /-
      🎉 no goals
    -/


protected theorem mul_comm : I * J = J * I :=
  le_antisymm (mul_le.2 fun _ hrI _ hsJ => mul_mem_mul_rev hsJ hrI)
    (mul_le.2 fun _ hrJ _ hsI => mul_mem_mul_rev hsI hrJ)


theorem span_mul_span (S T : Set R) : span S * span T = span (⋃ (s ∈ S) (t ∈ T), {s * t}) :=
  Submodule.span_smul_span S T


theorem span_mul_span' (S T : Set R) : span S * span T = span (S * T) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    S T : Set R
    ⊢ Eq (HMul.hMul (Ideal.span S) (Ideal.span T)) (Ideal.span (HMul.hMul S T))
  -/
  unfold span
  /-
    R : Type u
    inst✝ : CommSemiring R
    S T : Set R
    ⊢ Eq (HMul.hMul (Submodule.span R S) (Submodule.span R T)) (Submodule.span R ( …
  -/
  rw [Submodule.span_mul_span]
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_span_singleton (r s : R) :
    span {r} * span {s} = (span {r * s} : Ideal R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    r s : R
    ⊢ Eq (HMul.hMul (Ideal.span (Singleton.singleton r)) (Ideal.span (Singleton.si …
  -/
  unfold span
  /-
    R : Type u
    inst✝ : CommSemiring R
    r s : R
    ⊢ Eq (HMul.hMul (Submodule.span R (Singleton.singleton r)) (Submodule.span R ( …
  -/
  rw [Submodule.span_mul_span, Set.singleton_mul_singleton]
  /-
    🎉 no goals
  -/


theorem span_singleton_pow (s : R) (n : ℕ) : span {s} ^ n = (span {s ^ n} : Ideal R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : R
    n : Nat
    ⊢ Eq (HPow.hPow (Ideal.span (Singleton.singleton s)) n) (Ideal.span (Singleton …
  -/
  induction' n with n ih; · simp [Set.singleton_one]
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    R : Type u
    inst✝ : CommSemiring R
    s : R
    n : Nat
    ih : Eq (HPow.hPow (Ideal.span (Singleton.singleton s)) n) (Ideal.span (Single …
    ⊢ Eq (HPow.hPow (Ideal.span (Singleton.singleton s)) (HAdd.hAdd n 1)) (Ideal.s …
  -/
  simp only [pow_succ, ih, span_singleton_mul_span_singleton]
  /-
    🎉 no goals
  -/


theorem mem_mul_span_singleton {x y : R} {I : Ideal R} : x ∈ I * span {y} ↔ ∃ z ∈ I, z * y = x :=
  Submodule.mem_smul_span_singleton


theorem mem_span_singleton_mul {x y : R} {I : Ideal R} : x ∈ span {y} * I ↔ ∃ z ∈ I, y * z = x := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    I : Ideal R
    ⊢ Iff (Membership.mem (HMul.hMul (Ideal.span (Singleton.singleton y)) I) x) (E …
  -/
  simp only [mul_comm, mem_mul_span_singleton]
  /-
    🎉 no goals
  -/


theorem le_span_singleton_mul_iff {x : R} {I J : Ideal R} :
    I ≤ span {x} * J ↔ ∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI :=
  show (∀ {zI} (_ : zI ∈ I), zI ∈ span {x} * J) ↔ ∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI by
    /-
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      ⊢ Iff (∀ {zI : R}, Membership.mem I zI → Membership.mem (HMul.hMul (Ideal.span …
    -/
    simp only [mem_span_singleton_mul]
    /-
      🎉 no goals
    -/


theorem span_singleton_mul_le_iff {x : R} {I J : Ideal R} :
    span {x} * I ≤ J ↔ ∀ z ∈ I, x * z ∈ J := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    I J : Ideal R
    ⊢ Iff (LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) I) J) (∀ (z : R), …
  -/
  simp only [mul_le, mem_span_singleton_mul, mem_span_singleton]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    I J : Ideal R
    ⊢ Iff (∀ (r : R), Dvd.dvd x r → ∀ (s : R), Membership.mem I s → Membership.mem …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      ⊢ (∀ (r : R), Dvd.dvd x r → ∀ (s : R), Membership.mem I s → Membership.mem J ( …
    -/
  · intro h zI hzI
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      h : ∀ (r : R), Dvd.dvd x r → ∀ (s : R), Membership.mem I s → Membership.mem J  …
      zI : R
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HMul.hMul x zI)
    -/
    exact h x (dvd_refl x) zI hzI
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      ⊢ (∀ (z : R), Membership.mem I z → Membership.mem J (HMul.hMul x z)) → ∀ (r :  …
    -/
  · rintro h _ ⟨z, rfl⟩ zI hzI
    /-
      case mpr.intro
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      h : ∀ (z : R), Membership.mem I z → Membership.mem J (HMul.hMul x z)
      z zI : R
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HMul.hMul (HMul.hMul x z) zI)
    -/
    rw [mul_comm x z, mul_assoc]
    /-
      case mpr.intro
      R : Type u
      inst✝ : CommSemiring R
      x : R
      I J : Ideal R
      h : ∀ (z : R), Membership.mem I z → Membership.mem J (HMul.hMul x z)
      z zI : R
      hzI : Membership.mem I zI
      ⊢ Membership.mem J (HMul.hMul z (HMul.hMul x zI))
    -/
    exact J.mul_mem_left _ (h zI hzI)
    /-
      🎉 no goals
    -/


theorem span_singleton_mul_le_span_singleton_mul {x y : R} {I J : Ideal R} :
    span {x} * I ≤ span {y} * J ↔ ∀ zI ∈ I, ∃ zJ ∈ J, x * zI = y * zJ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    I J : Ideal R
    ⊢ Iff (LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) I) (HMul.hMul (Id …
  -/
  simp only [span_singleton_mul_le_iff, mem_span_singleton_mul, eq_comm]
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_right_mono [IsDomain R] {x : R} (hx : x ≠ 0) :
    span {x} * I ≤ span {x} * J ↔ I ≤ J := by
  simp_rw [span_singleton_mul_le_span_singleton_mul, mul_right_inj' hx,
    exists_eq_right', SetLike.le_def]


theorem span_singleton_mul_left_mono [IsDomain R] {x : R} (hx : x ≠ 0) :
    I * span {x} ≤ J * span {x} ↔ I ≤ J := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    ⊢ Iff (LE.le (HMul.hMul I (Ideal.span (Singleton.singleton x))) (HMul.hMul J ( …
  -/
  simpa only [mul_comm I, mul_comm J] using span_singleton_mul_right_mono hx
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_right_inj [IsDomain R] {x : R} (hx : x ≠ 0) :
    span {x} * I = span {x} * J ↔ I = J := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    ⊢ Iff (Eq (HMul.hMul (Ideal.span (Singleton.singleton x)) I) (HMul.hMul (Ideal …
  -/
  simp only [le_antisymm_iff, span_singleton_mul_right_mono hx]
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_left_inj [IsDomain R] {x : R} (hx : x ≠ 0) :
    I * span {x} = J * span {x} ↔ I = J := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    I J : Ideal R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    ⊢ Iff (Eq (HMul.hMul I (Ideal.span (Singleton.singleton x))) (HMul.hMul J (Ide …
  -/
  simp only [le_antisymm_iff, span_singleton_mul_left_mono hx]
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_right_injective [IsDomain R] {x : R} (hx : x ≠ 0) :
    Function.Injective ((span {x} : Ideal R) * ·) := fun _ _ =>
  (span_singleton_mul_right_inj hx).mp


theorem span_singleton_mul_left_injective [IsDomain R] {x : R} (hx : x ≠ 0) :
    Function.Injective fun I : Ideal R => I * span {x} := fun _ _ =>
  (span_singleton_mul_left_inj hx).mp


theorem eq_span_singleton_mul {x : R} (I J : Ideal R) :
    I = span {x} * J ↔ (∀ zI ∈ I, ∃ zJ ∈ J, x * zJ = zI) ∧ ∀ z ∈ J, x * z ∈ I := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    I J : Ideal R
    ⊢ Iff (Eq I (HMul.hMul (Ideal.span (Singleton.singleton x)) J)) (And (∀ (zI :  …
  -/
  simp only [le_antisymm_iff, le_span_singleton_mul_iff, span_singleton_mul_le_iff]
  /-
    🎉 no goals
  -/


theorem span_singleton_mul_eq_span_singleton_mul {x y : R} (I J : Ideal R) :
    span {x} * I = span {y} * J ↔
      (∀ zI ∈ I, ∃ zJ ∈ J, x * zI = y * zJ) ∧ ∀ zJ ∈ J, ∃ zI ∈ I, x * zI = y * zJ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    I J : Ideal R
    ⊢ Iff (Eq (HMul.hMul (Ideal.span (Singleton.singleton x)) I) (HMul.hMul (Ideal …
  -/
  simp only [le_antisymm_iff, span_singleton_mul_le_span_singleton_mul, eq_comm]
  /-
    🎉 no goals
  -/


theorem prod_span {ι : Type*} (s : Finset ι) (I : ι → Set R) :
    (∏ i ∈ s, Ideal.span (I i)) = Ideal.span (∏ i ∈ s, I i) :=
  Submodule.prod_span s I


theorem prod_span_singleton {ι : Type*} (s : Finset ι) (I : ι → R) :
    (∏ i ∈ s, Ideal.span ({I i} : Set R)) = Ideal.span {∏ i ∈ s, I i} :=
  Submodule.prod_span_singleton s I


@[simp]
theorem multiset_prod_span_singleton (m : Multiset R) :
    (m.map fun x => Ideal.span {x}).prod = Ideal.span ({Multiset.prod m} : Set R) :=
                              /-
                                R : Type u
                                inst✝ : CommSemiring R
                                m : Multiset R
                                ⊢ Eq (Multiset.map (fun x => Ideal.span (Singleton.singleton x)) 0).prod (Idea …
                              -/
  Multiset.induction_on m (by simp) fun a m ih => by
                              /-
                                🎉 no goals
                              -/
    /-
      R : Type u
      inst✝ : CommSemiring R
      m✝ : Multiset R
      a : R
      m : Multiset R
      ih : Eq (Multiset.map (fun x => Ideal.span (Singleton.singleton x)) m).prod (I …
      ⊢ Eq (Multiset.map (fun x => Ideal.span (Singleton.singleton x)) (Multiset.con …
    -/
    simp only [Multiset.map_cons, Multiset.prod_cons, ih, ← Ideal.span_singleton_mul_span_singleton]
    /-
      🎉 no goals
    -/


theorem finset_inf_span_singleton {ι : Type*} (s : Finset ι) (I : ι → R)
    (hI : Set.Pairwise (↑s) (IsCoprime on I)) :
    (s.inf fun i => Ideal.span ({I i} : Set R)) = Ideal.span {∏ i ∈ s, I i} := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_2
    s : Finset ι
    I : ι → R
    hI : (↑s).Pairwise (Function.onFun IsCoprime I)
    ⊢ Eq (s.inf fun i => Ideal.span (Singleton.singleton (I i))) (Ideal.span (Sing …
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_2
    s : Finset ι
    I : ι → R
    hI : (↑s).Pairwise (Function.onFun IsCoprime I)
    x : R
    ⊢ Iff (Membership.mem (s.inf fun i => Ideal.span (Singleton.singleton (I i)))  …
  -/
  simp only [Submodule.mem_finset_inf, Ideal.mem_span_singleton]
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_2
    s : Finset ι
    I : ι → R
    hI : (↑s).Pairwise (Function.onFun IsCoprime I)
    x : R
    ⊢ Iff (∀ (i : ι), Membership.mem s i → Dvd.dvd (I i) x) (Dvd.dvd (s.prod fun i …
  -/
  exact ⟨Finset.prod_dvd_of_coprime hI, fun h i hi => (Finset.dvd_prod_of_mem _ hi).trans h⟩
  /-
    🎉 no goals
  -/


theorem iInf_span_singleton {ι : Type*} [Fintype ι] {I : ι → R}
    (hI : ∀ (i j) (_ : i ≠ j), IsCoprime (I i) (I j)) :
    ⨅ i, span ({I i} : Set R) = span {∏ i, I i} := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    ι : Type u_2
    inst✝ : Fintype ι
    I : ι → R
    hI : ∀ (i j : ι), Ne i j → IsCoprime (I i) (I j)
    ⊢ Eq (iInf fun i => Ideal.span (Singleton.singleton (I i))) (Ideal.span (Singl …
  -/
  rw [← Finset.inf_univ_eq_iInf, finset_inf_span_singleton]
  /-
    case hI
    R : Type u
    inst✝¹ : CommSemiring R
    ι : Type u_2
    inst✝ : Fintype ι
    I : ι → R
    hI : ∀ (i j : ι), Ne i j → IsCoprime (I i) (I j)
    ⊢ (↑Finset.univ).Pairwise (Function.onFun IsCoprime I)
  -/
  rwa [Finset.coe_univ, Set.pairwise_univ]
  /-
    🎉 no goals
  -/


theorem iInf_span_singleton_natCast {R : Type*} [CommRing R] {ι : Type*} [Fintype ι]
    {I : ι → ℕ} (hI : Pairwise fun i j => (I i).Coprime (I j)) :
    ⨅ (i : ι), span {(I i : R)} = span {((∏ i : ι, I i : ℕ) : R)} := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Fintype ι
    I : ι → Nat
    hI : Pairwise fun i j => (I i).Coprime (I j)
    ⊢ Eq (iInf fun i => Ideal.span (Singleton.singleton ↑(I i))) (Ideal.span (Sing …
  -/
  rw [iInf_span_singleton, Nat.cast_prod]
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Fintype ι
    I : ι → Nat
    hI : Pairwise fun i j => (I i).Coprime (I j)
    ⊢ ∀ (i j : ι), Ne i j → IsCoprime ↑(I i) ↑(I j)
  -/
  exact fun i j h ↦ (hI h).cast
  /-
    🎉 no goals
  -/


theorem sup_eq_top_iff_isCoprime {R : Type*} [CommSemiring R] (x y : R) :
    span ({x} : Set R) ⊔ span {y} = ⊤ ↔ IsCoprime x y := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    x y : R
    ⊢ Iff (Eq (Max.max (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton …
  -/
  rw [eq_top_iff_one, Submodule.mem_sup]
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    x y : R
    ⊢ Iff (Exists fun y_1 => And (Membership.mem (Ideal.span (Singleton.singleton  …
  -/
  constructor
    /-
      case mp
      R : Type u_2
      inst✝ : CommSemiring R
      x y : R
      ⊢ (Exists fun y_1 => And (Membership.mem (Ideal.span (Singleton.singleton x))  …
    -/
  · rintro ⟨u, hu, v, hv, h1⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_2
      inst✝ : CommSemiring R
      x y u : R
      hu : Membership.mem (Ideal.span (Singleton.singleton x)) u
      v : R
      hv : Membership.mem (Ideal.span (Singleton.singleton y)) v
      h1 : Eq (HAdd.hAdd u v) 1
      ⊢ IsCoprime x y
    -/
    rw [mem_span_singleton'] at hu hv
    /-
      case mp.intro.intro.intro.intro
      R : Type u_2
      inst✝ : CommSemiring R
      x y u : R
      hu : Exists fun a => Eq (HMul.hMul a x) u
      v : R
      hv : Exists fun a => Eq (HMul.hMul a y) v
      h1 : Eq (HAdd.hAdd u v) 1
      ⊢ IsCoprime x y
    -/
    rw [← hu.choose_spec, ← hv.choose_spec] at h1
    /-
      case mp.intro.intro.intro.intro
      R : Type u_2
      inst✝ : CommSemiring R
      x y u : R
      hu : Exists fun a => Eq (HMul.hMul a x) u
      v : R
      hv : Exists fun a => Eq (HMul.hMul a y) v
      h1 : Eq (HAdd.hAdd (HMul.hMul hu.choose x) (HMul.hMul hv.choose y)) 1
      ⊢ IsCoprime x y
    -/
    exact ⟨_, _, h1⟩
    /-
      🎉 no goals
    -/
  · exact fun ⟨u, v, h1⟩ =>
      ⟨_, mem_span_singleton'.mpr ⟨_, rfl⟩, _, mem_span_singleton'.mpr ⟨_, rfl⟩, h1⟩


theorem mul_le_inf : I * J ≤ I ⊓ J :=
  mul_le.2 fun r hri s hsj => ⟨I.mul_mem_right s hri, J.mul_mem_left r hsj⟩


theorem multiset_prod_le_inf {s : Multiset (Ideal R)} : s.prod ≤ s.inf := by
  classical
    refine s.induction_on ?_ ?_
    · rw [Multiset.inf_zero]
      exact le_top
    intro a s ih
    rw [Multiset.prod_cons, Multiset.inf_cons]
    exact le_trans mul_le_inf (inf_le_inf le_rfl ih)


theorem prod_le_inf {s : Finset ι} {f : ι → Ideal R} : s.prod f ≤ s.inf f :=
  multiset_prod_le_inf


theorem mul_eq_inf_of_coprime (h : I ⊔ J = ⊤) : I * J = I ⊓ J :=
  le_antisymm mul_le_inf fun r ⟨hri, hrj⟩ =>
    let ⟨s, hsi, t, htj, hst⟩ := Submodule.mem_sup.1 ((eq_top_iff_one _).1 h)
    mul_one r ▸
      hst ▸
        (mul_add r s t).symm ▸ Ideal.add_mem (I * J) (mul_mem_mul_rev hsi hrj) (mul_mem_mul hri htj)


theorem sup_mul_eq_of_coprime_left (h : I ⊔ J = ⊤) : I ⊔ J * K = I ⊔ K :=
  le_antisymm (sup_le_sup_left mul_le_left _) fun i hi => by
    /-
      R : Type u
      inst✝ : CommSemiring R
      I J K : Ideal R
      h : Eq (Max.max I J) Top.top
      i : R
      hi : Membership.mem (Max.max I K) i
      ⊢ Membership.mem (Max.max I (HMul.hMul J K)) i
    -/
    rw [eq_top_iff_one] at h; rw [Submodule.mem_sup] at h hi ⊢
    /-
      R : Type u
      inst✝ : CommSemiring R
      I J K : Ideal R
      h : Exists fun y => And (Membership.mem I y) (Exists fun z => And (Membership. …
      i : R
      hi : Exists fun y => And (Membership.mem I y) (Exists fun z => And (Membership …
      ⊢ Exists fun y => And (Membership.mem I y) (Exists fun z => And (Membership.me …
    -/
    obtain ⟨i1, hi1, j, hj, h⟩ := h; obtain ⟨i', hi', k, hk, hi⟩ := hi
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I J K : Ideal R
      i i1 : R
      hi1 : Membership.mem I i1
      j : R
      hj : Membership.mem J j
      h : Eq (HAdd.hAdd i1 j) 1
      i' : R
      hi' : Membership.mem I i'
      k : R
      hk : Membership.mem K k
      hi : Eq (HAdd.hAdd i' k) i
      ⊢ Exists fun y => And (Membership.mem I y) (Exists fun z => And (Membership.me …
    -/
    refine ⟨_, add_mem hi' (mul_mem_right k _ hi1), _, mul_mem_mul hj hk, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I J K : Ideal R
      i i1 : R
      hi1 : Membership.mem I i1
      j : R
      hj : Membership.mem J j
      h : Eq (HAdd.hAdd i1 j) 1
      i' : R
      hi' : Membership.mem I i'
      k : R
      hk : Membership.mem K k
      hi : Eq (HAdd.hAdd i' k) i
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd i' (HMul.hMul i1 k)) (HMul.hMul j k)) i
    -/
    rw [add_assoc, ← add_mul, h, one_mul, hi]
    /-
      🎉 no goals
    -/


theorem sup_mul_eq_of_coprime_right (h : I ⊔ K = ⊤) : I ⊔ J * K = I ⊔ J := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max I K) Top.top
    ⊢ Eq (Max.max I (HMul.hMul J K)) (Max.max I J)
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max I K) Top.top
    ⊢ Eq (Max.max I (HMul.hMul K J)) (Max.max I J)
  -/
  exact sup_mul_eq_of_coprime_left h
  /-
    🎉 no goals
  -/


theorem mul_sup_eq_of_coprime_left (h : I ⊔ J = ⊤) : I * K ⊔ J = K ⊔ J := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max I J) Top.top
    ⊢ Eq (Max.max (HMul.hMul I K) J) (Max.max K J)
  -/
  rw [sup_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max J I) Top.top
    ⊢ Eq (Max.max (HMul.hMul I K) J) (Max.max K J)
  -/
  rw [sup_comm, sup_mul_eq_of_coprime_left h, sup_comm]
  /-
    🎉 no goals
  -/


theorem mul_sup_eq_of_coprime_right (h : K ⊔ J = ⊤) : I * K ⊔ J = I ⊔ J := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max K J) Top.top
    ⊢ Eq (Max.max (HMul.hMul I K) J) (Max.max I J)
  -/
  rw [sup_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J K : Ideal R
    h : Eq (Max.max J K) Top.top
    ⊢ Eq (Max.max (HMul.hMul I K) J) (Max.max I J)
  -/
  rw [sup_comm, sup_mul_eq_of_coprime_right h, sup_comm]
  /-
    🎉 no goals
  -/


theorem sup_prod_eq_top {s : Finset ι} {J : ι → Ideal R} (h : ∀ i, i ∈ s → I ⊔ J i = ⊤) :
    (I ⊔ ∏ i ∈ s, J i) = ⊤ :=
  Finset.prod_induction _ (fun J => I ⊔ J = ⊤)
    (fun _ _ hJ hK => (sup_mul_eq_of_coprime_left hJ).trans hK)
        /-
          R : Type u
          ι : Type u_1
          inst✝ : CommSemiring R
          I : Ideal R
          s : Finset ι
          J : ι → Ideal R
          h : ∀ (i : ι), Membership.mem s i → Eq (Max.max I (J i)) Top.top
          ⊢ (fun J => Eq (Max.max I J) Top.top) 1
        -/
    (by simp_rw [one_eq_top, sup_top_eq]) h
        /-
          🎉 no goals
        -/


theorem sup_multiset_prod_eq_top {s : Multiset (Ideal R)} (h : ∀  p ∈ s, I ⊔ p = ⊤) :
    I ⊔ Multiset.prod s = ⊤ :=
  Multiset.prod_induction (I ⊔ · = ⊤) s (fun _ _ hp hq ↦ (sup_mul_eq_of_coprime_left hp).trans hq)
        /-
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          s : Multiset (Ideal R)
          h : ∀ (p : Ideal R), Membership.mem s p → Eq (Max.max I p) Top.top
          ⊢ (fun x => Eq (Max.max I x) Top.top) 1
        -/
    (by simp only [one_eq_top, ge_iff_le, top_le_iff, le_top, sup_of_le_right]) h
        /-
          🎉 no goals
        -/


theorem sup_iInf_eq_top {s : Finset ι} {J : ι → Ideal R} (h : ∀ i, i ∈ s → I ⊔ J i = ⊤) :
    (I ⊔ ⨅ i ∈ s, J i) = ⊤ :=
  eq_top_iff.mpr <|
    le_of_eq_of_le (sup_prod_eq_top h).symm <|
      sup_le_sup_left (le_of_le_of_eq prod_le_inf <| Finset.inf_eq_iInf _ _) _


theorem prod_sup_eq_top {s : Finset ι} {J : ι → Ideal R} (h : ∀ i, i ∈ s → J i ⊔ I = ⊤) :
                                 /-
                                   R : Type u
                                   ι : Type u_1
                                   inst✝ : CommSemiring R
                                   I : Ideal R
                                   s : Finset ι
                                   J : ι → Ideal R
                                   h : ∀ (i : ι), Membership.mem s i → Eq (Max.max (J i) I) Top.top
                                   ⊢ Eq (Max.max (s.prod fun i => J i) I) Top.top
                                 -/
    (∏ i ∈ s, J i) ⊔ I = ⊤ := by rw [sup_comm, sup_prod_eq_top]; intro i hi; rw [sup_comm, h i hi]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem iInf_sup_eq_top {s : Finset ι} {J : ι → Ideal R} (h : ∀ i, i ∈ s → J i ⊔ I = ⊤) :
                                 /-
                                   R : Type u
                                   ι : Type u_1
                                   inst✝ : CommSemiring R
                                   I : Ideal R
                                   s : Finset ι
                                   J : ι → Ideal R
                                   h : ∀ (i : ι), Membership.mem s i → Eq (Max.max (J i) I) Top.top
                                   ⊢ Eq (Max.max (iInf fun i => iInf fun h => J i) I) Top.top
                                 -/
    (⨅ i ∈ s, J i) ⊔ I = ⊤ := by rw [sup_comm, sup_iInf_eq_top]; intro i hi; rw [sup_comm, h i hi]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem sup_pow_eq_top {n : ℕ} (h : I ⊔ J = ⊤) : I ⊔ J ^ n = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n : Nat
    h : Eq (Max.max I J) Top.top
    ⊢ Eq (Max.max I (HPow.hPow J n)) Top.top
  -/
  rw [← Finset.card_range n, ← Finset.prod_const]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n : Nat
    h : Eq (Max.max I J) Top.top
    ⊢ Eq (Max.max I ((Finset.range n).prod fun _x => J)) Top.top
  -/
  exact sup_prod_eq_top fun _ _ => h
  /-
    🎉 no goals
  -/


theorem pow_sup_eq_top {n : ℕ} (h : I ⊔ J = ⊤) : I ^ n ⊔ J = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n : Nat
    h : Eq (Max.max I J) Top.top
    ⊢ Eq (Max.max (HPow.hPow I n) J) Top.top
  -/
  rw [← Finset.card_range n, ← Finset.prod_const]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    n : Nat
    h : Eq (Max.max I J) Top.top
    ⊢ Eq (Max.max ((Finset.range n).prod fun _x => I) J) Top.top
  -/
  exact prod_sup_eq_top fun _ _ => h
  /-
    🎉 no goals
  -/


theorem pow_sup_pow_eq_top {m n : ℕ} (h : I ⊔ J = ⊤) : I ^ m ⊔ J ^ n = ⊤ :=
  sup_pow_eq_top (pow_sup_eq_top h)


variable (I) in
@[simp]
theorem mul_top : I * ⊤ = I :=
  Ideal.mul_comm ⊤ I ▸ Submodule.top_smul I


/-- A product of ideals in an integral domain is zero if and only if one of the terms is zero. -/
@[simp]
lemma multiset_prod_eq_bot {R : Type*} [CommRing R] [IsDomain R] {s : Multiset (Ideal R)} :
    s.prod = ⊥ ↔ ⊥ ∈ s :=
  Multiset.prod_eq_zero_iff


theorem span_pair_mul_span_pair (w x y z : R) :
    (span {w, x} : Ideal R) * span {y, z} = span {w * y, w * z, x * y, x * z} := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    w x y z : R
    ⊢ Eq (HMul.hMul (Ideal.span (Insert.insert w (Singleton.singleton x))) (Ideal. …
  -/
  simp_rw [span_insert, sup_mul, mul_sup, span_singleton_mul_span_singleton, sup_assoc]
  /-
    🎉 no goals
  -/


theorem isCoprime_iff_codisjoint : IsCoprime I J ↔ Codisjoint I J := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (IsCoprime I J) (Codisjoint I J)
  -/
  rw [IsCoprime, codisjoint_iff]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a I) (HMul.hMu …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      ⊢ (Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a I) (HMul.hMul b  …
    -/
  · rintro ⟨x, y, hxy⟩
    /-
      case mp.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I J x y : Ideal R
      hxy : Eq (HAdd.hAdd (HMul.hMul x I) (HMul.hMul y J)) 1
      ⊢ Eq (Max.max I J) Top.top
    -/
    rw [eq_top_iff_one]
    apply (show x * I + y * J ≤ I ⊔ J from
      sup_le (mul_le_left.trans le_sup_left) (mul_le_left.trans le_sup_right))
    /-
      case mp.intro.intro.a
      R : Type u
      inst✝ : CommSemiring R
      I J x y : Ideal R
      hxy : Eq (HAdd.hAdd (HMul.hMul x I) (HMul.hMul y J)) 1
      ⊢ Membership.mem (HAdd.hAdd (HMul.hMul x I) (HMul.hMul y J)) 1
    -/
    rw [hxy]
    /-
      case mp.intro.intro.a
      R : Type u
      inst✝ : CommSemiring R
      I J x y : Ideal R
      hxy : Eq (HAdd.hAdd (HMul.hMul x I) (HMul.hMul y J)) 1
      ⊢ Membership.mem 1 1
    -/
    simp only [one_eq_top, Submodule.mem_top]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      ⊢ Eq (Max.max I J) Top.top → Exists fun a => Exists fun b => Eq (HAdd.hAdd (HM …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      h : Eq (Max.max I J) Top.top
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a I) (HMul.hMul b J …
    -/
    refine ⟨1, 1, ?_⟩
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      h : Eq (Max.max I J) Top.top
      ⊢ Eq (HAdd.hAdd (HMul.hMul 1 I) (HMul.hMul 1 J)) 1
    -/
    simpa only [one_eq_top, top_mul, Submodule.add_eq_sup]
    /-
      🎉 no goals
    -/


theorem isCoprime_iff_add : IsCoprime I J ↔ I + J = 1 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (IsCoprime I J) (Eq (HAdd.hAdd I J) 1)
  -/
  rw [isCoprime_iff_codisjoint, codisjoint_iff, add_eq_sup, one_eq_top]
  /-
    🎉 no goals
  -/


theorem isCoprime_iff_exists : IsCoprime I J ↔ ∃ i ∈ I, ∃ j ∈ J, i + j = 1 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (IsCoprime I J) (Exists fun i => And (Membership.mem I i) (Exists fun j  …
  -/
  rw [← add_eq_one_iff, isCoprime_iff_add]
  /-
    🎉 no goals
  -/


theorem isCoprime_iff_sup_eq : IsCoprime I J ↔ I ⊔ J = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (IsCoprime I J) (Eq (Max.max I J) Top.top)
  -/
  rw [isCoprime_iff_codisjoint, codisjoint_iff]
  /-
    🎉 no goals
  -/


open List in
theorem isCoprime_tfae : TFAE [IsCoprime I J, Codisjoint I J, I + J = 1,
    ∃ i ∈ I, ∃ j ∈ J, i + j = 1, I ⊔ J = ⊤] := by
  rw [← isCoprime_iff_codisjoint, ← isCoprime_iff_add, ← isCoprime_iff_exists,
      ← isCoprime_iff_sup_eq]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ (List.cons (IsCoprime I J) (List.cons (IsCoprime I J) (List.cons (IsCoprime  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem _root_.IsCoprime.codisjoint (h : IsCoprime I J) : Codisjoint I J :=
  isCoprime_iff_codisjoint.mp h


theorem _root_.IsCoprime.add_eq (h : IsCoprime I J) : I + J = 1 := isCoprime_iff_add.mp h


theorem _root_.IsCoprime.exists (h : IsCoprime I J) : ∃ i ∈ I, ∃ j ∈ J, i + j = 1 :=
  isCoprime_iff_exists.mp h


theorem _root_.IsCoprime.sup_eq (h : IsCoprime I J) : I ⊔ J = ⊤ := isCoprime_iff_sup_eq.mp h


theorem inf_eq_mul_of_isCoprime (coprime : IsCoprime I J) : I ⊓ J = I * J :=
  (Ideal.mul_eq_inf_of_coprime coprime.sup_eq).symm


@[deprecated (since := "2024-05-28")]
alias inf_eq_mul_of_coprime := inf_eq_mul_of_isCoprime


theorem isCoprime_span_singleton_iff (x y : R) :
    IsCoprime (span <| singleton x) (span <| singleton y) ↔ IsCoprime x y := by
  simp_rw [isCoprime_iff_codisjoint, codisjoint_iff, eq_top_iff_one, mem_span_singleton_sup,
    mem_span_singleton]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    ⊢ Iff (Exists fun a => Exists fun b => And (Dvd.dvd y b) (Eq (HAdd.hAdd (HMul. …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      x y : R
      ⊢ (Exists fun a => Exists fun b => And (Dvd.dvd y b) (Eq (HAdd.hAdd (HMul.hMul …
    -/
  · rintro ⟨a, _, ⟨b, rfl⟩, e⟩; exact ⟨a, b, mul_comm b y ▸ e⟩
                                /-
                                  🎉 no goals
                                -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      x y : R
      ⊢ IsCoprime x y → Exists fun a => Exists fun b => And (Dvd.dvd y b) (Eq (HAdd. …
    -/
  · rintro ⟨a, b, e⟩; exact ⟨a, _, ⟨b, rfl⟩, mul_comm y b ▸ e⟩
                      /-
                        🎉 no goals
                      -/


theorem isCoprime_biInf {J : ι → Ideal R} {s : Finset ι}
    (hf : ∀ j ∈ s, IsCoprime I (J j)) : IsCoprime I (⨅ j ∈ s, J j) := by
  classical
  simp_rw [isCoprime_iff_add] at *
  induction s using Finset.induction with
  | empty =>
      simp
  | @insert i s _ hs =>
      rw [Finset.iInf_insert, inf_comm, one_eq_top, eq_top_iff, ← one_eq_top]
      set K := ⨅ j ∈ s, J j
      calc
        1 = I + K            := (hs fun j hj ↦ hf j (Finset.mem_insert_of_mem hj)).symm
        _ = I + K*(I + J i)  := by rw [hf i (Finset.mem_insert_self i s), mul_one]
        _ = (1+K)*I + K*J i  := by ring
        _ ≤ I + K ⊓ J i      := add_le_add mul_le_left mul_le_inf


/-- The radical of an ideal `I` consists of the elements `r` such that `r ^ n ∈ I` for some `n`. -/
def radical (I : Ideal R) : Ideal R where
  carrier := { r | ∃ n : ℕ, r ^ n ∈ I }
  zero_mem' := ⟨1, (pow_one (0 : R)).symm ▸ I.zero_mem⟩
  add_mem' := fun {_ _} ⟨m, hxmi⟩ ⟨n, hyni⟩ =>
    ⟨m + n - 1, add_pow_add_pred_mem_of_pow_mem I hxmi hyni⟩
  smul_mem' {r s} := fun ⟨n, h⟩ ↦ ⟨n, (mul_pow r s n).symm ▸ I.mul_mem_left (r ^ n) h⟩


theorem mem_radical_iff {r : R} : r ∈ I.radical ↔ ∃ n : ℕ, r ^ n ∈ I := Iff.rfl


/-- An ideal is radical if it contains its radical. -/
def IsRadical (I : Ideal R) : Prop :=
  I.radical ≤ I


theorem le_radical : I ≤ radical I := fun r hri => ⟨1, (pow_one r).symm ▸ hri⟩


/-- An ideal is radical iff it is equal to its radical. -/
theorem radical_eq_iff : I.radical = I ↔ I.IsRadical := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff (Eq I.radical I) I.IsRadical
  -/
  rw [le_antisymm_iff, and_iff_left le_radical, IsRadical]
  /-
    🎉 no goals
  -/


alias ⟨_, IsRadical.radical⟩ := radical_eq_iff


theorem isRadical_iff_pow_one_lt (k : ℕ) (hk : 1 < k) : I.IsRadical ↔ ∀ r, r ^ k ∈ I → r ∈ I :=
  ⟨fun h _r hr ↦ h ⟨k, hr⟩, fun h x ⟨n, hx⟩ ↦
    k.pow_imp_self_of_one_lt hk _ (fun _ _ ↦ .inr ∘ I.smul_mem _) h n x hx⟩


theorem radical_top : (radical ⊤ : Ideal R) = ⊤ :=
  (eq_top_iff_one _).2 ⟨0, Submodule.mem_top⟩


theorem radical_mono (H : I ≤ J) : radical I ≤ radical J := fun _ ⟨n, hrni⟩ => ⟨n, H hrni⟩


theorem radical_isRadical : (radical I).IsRadical := fun r ⟨n, k, hrnki⟩ =>
  ⟨n * k, (pow_mul r n k).symm ▸ hrnki⟩


@[simp]
theorem radical_idem : radical (radical I) = radical I :=
  (radical_isRadical I).radical


theorem IsRadical.radical_le_iff (hJ : J.IsRadical) : I.radical ≤ J ↔ I ≤ J :=
  ⟨le_trans le_radical, fun h => hJ.radical ▸ radical_mono h⟩


theorem radical_le_radical_iff : radical I ≤ radical J ↔ I ≤ radical J :=
  (radical_isRadical J).radical_le_iff


theorem radical_eq_top : radical I = ⊤ ↔ I = ⊤ :=
  ⟨fun h =>
    (eq_top_iff_one _).2 <|
      let ⟨n, hn⟩ := (eq_top_iff_one _).1 h
      @one_pow R _ n ▸ hn,
    fun h => h.symm ▸ radical_top R⟩


theorem IsPrime.isRadical (H : IsPrime I) : I.IsRadical := fun _ ⟨n, hrni⟩ =>
  H.mem_of_pow_mem n hrni


theorem IsPrime.radical (H : IsPrime I) : radical I = I :=
  IsRadical.radical H.isRadical


theorem mem_radical_of_pow_mem {I : Ideal R} {x : R} {m : ℕ} (hx : x ^ m ∈ radical I) :
    x ∈ radical I :=
  radical_idem I ▸ ⟨m, hx⟩


theorem disjoint_powers_iff_not_mem (y : R) (hI : I.IsRadical) :
    Disjoint (Submonoid.powers y : Set R) ↑I ↔ y ∉ I.1 := by
  refine ⟨fun h => Set.disjoint_left.1 h (Submonoid.mem_powers _),
      fun h => disjoint_iff.mpr (eq_bot_iff.mpr ?_)⟩
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    y : R
    hI : I.IsRadical
    h : Not (Membership.mem I.toAddSubmonoid y)
    ⊢ LE.le (Min.min ↑(Submonoid.powers y) ↑I) Bot.bot
  -/
  rintro x ⟨⟨n, rfl⟩, hx'⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    y : R
    hI : I.IsRadical
    h : Not (Membership.mem I.toAddSubmonoid y)
    n : Nat
    hx' : Membership.mem (↑I) ((fun x => HPow.hPow y x) n)
    ⊢ Membership.mem Bot.bot ((fun x => HPow.hPow y x) n)
  -/
  exact h (hI <| mem_radical_of_pow_mem <| le_radical hx')
  /-
    🎉 no goals
  -/


theorem radical_sup : radical (I ⊔ J) = radical (radical I ⊔ radical J) :=
  le_antisymm (radical_mono <| sup_le_sup le_radical le_radical) <|
    radical_le_radical_iff.2 <| sup_le (radical_mono le_sup_left) (radical_mono le_sup_right)


theorem radical_inf : radical (I ⊓ J) = radical I ⊓ radical J :=
  le_antisymm (le_inf (radical_mono inf_le_left) (radical_mono inf_le_right))
    fun r ⟨⟨m, hrm⟩, ⟨n, hrn⟩⟩ =>
    ⟨m + n, (pow_add r m n).symm ▸ I.mul_mem_right _ hrm,
      (pow_add r m n).symm ▸ J.mul_mem_left _ hrn⟩


variable {I J} in
theorem IsRadical.inf (hI : IsRadical I) (hJ : IsRadical J) : IsRadical (I ⊓ J) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    hI : I.IsRadical
    hJ : J.IsRadical
    ⊢ (Min.min I J).IsRadical
  -/
  rw [IsRadical, radical_inf]; exact inf_le_inf hI hJ
                               /-
                                 🎉 no goals
                               -/


/-- `Ideal.radical` as an `InfTopHom`, bundling in that it distributes over `inf`. -/
def radicalInfTopHom : InfTopHom (Ideal R) (Ideal R) where
  toFun := radical
  map_inf' := radical_inf
  map_top' := radical_top _


@[simp]
lemma radicalInfTopHom_apply (I : Ideal R) : radicalInfTopHom I = radical I := rfl


open Finset in
lemma radical_finset_inf {ι} {s : Finset ι} {f : ι → Ideal R} {i : ι} (hi : i ∈ s)
    (hs : ∀ ⦃y⦄, y ∈ s → (f y).radical = (f i).radical) :
    (s.inf f).radical = (f i).radical := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_2
    s : Finset ι
    f : ι → Ideal R
    i : ι
    hi : Membership.mem s i
    hs : ∀ ⦃y : ι⦄, Membership.mem s y → Eq (f y).radical (f i).radical
    ⊢ Eq (s.inf f).radical (f i).radical
  -/
  rw [← radicalInfTopHom_apply, map_finset_inf, ← Finset.inf'_eq_inf ⟨_, hi⟩]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_2
    s : Finset ι
    f : ι → Ideal R
    i : ι
    hi : Membership.mem s i
    hs : ∀ ⦃y : ι⦄, Membership.mem s y → Eq (f y).radical (f i).radical
    ⊢ Eq (s.inf' ⋯ (Function.comp (⇑Ideal.radicalInfTopHom) f)) (f i).radical
  -/
  exact Finset.inf'_eq_of_forall _ _ hs
  /-
    🎉 no goals
  -/


/-- The reverse inclusion does not hold for e.g. `I := fun n : ℕ ↦ Ideal.span {(2 ^ n : ℤ)}`. -/
theorem radical_iInf_le {ι} (I : ι → Ideal R) : radical (⨅ i, I i) ≤ ⨅ i, radical (I i) :=
  le_iInf fun _ ↦ radical_mono (iInf_le _ _)


theorem isRadical_iInf {ι} (I : ι → Ideal R) (hI : ∀ i, IsRadical (I i)) : IsRadical (⨅ i, I i) :=
  (radical_iInf_le I).trans (iInf_mono hI)


theorem radical_mul : radical (I * J) = radical I ⊓ radical J := by
  refine le_antisymm ?_ fun r ⟨⟨m, hrm⟩, ⟨n, hrn⟩⟩ =>
    ⟨m + n, (pow_add r m n).symm ▸ mul_mem_mul hrm hrn⟩
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ LE.le (HMul.hMul I J).radical (Min.min I.radical J.radical)
  -/
  have := radical_mono <| @mul_le_inf _ _ I J
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    this : LE.le (HMul.hMul I J).radical (Min.min I J).radical
    ⊢ LE.le (HMul.hMul I J).radical (Min.min I.radical J.radical)
  -/
  simp_rw [radical_inf I J] at this
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    this : LE.le (HMul.hMul I J).radical (Min.min I.radical J.radical)
    ⊢ LE.le (HMul.hMul I J).radical (Min.min I.radical J.radical)
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem IsPrime.radical_le_iff (hJ : IsPrime J) : I.radical ≤ J ↔ I ≤ J :=
  IsRadical.radical_le_iff hJ.isRadical


theorem radical_eq_sInf (I : Ideal R) : radical I = sInf { J : Ideal R | I ≤ J ∧ IsPrime J } :=
  le_antisymm (le_sInf fun _ hJ ↦ hJ.2.radical_le_iff.2 hJ.1) fun r hr ↦
    by_contradiction fun hri ↦
      let ⟨m, hIm, hm⟩ :=
        zorn_le_nonempty₀ { K : Ideal R | r ∉ radical K }
          (fun c hc hcc y hyc =>
            ⟨sSup c, fun ⟨n, hrnc⟩ =>
              let ⟨_, hyc, hrny⟩ := (Submodule.mem_sSup_of_directed ⟨y, hyc⟩ hcc.directedOn).1 hrnc
              hc hyc ⟨n, hrny⟩,
              fun _ => le_sSup⟩)
          I hri
      have hrm : r ∉ radical m := hm.prop
      have : ∀ x ∉ m, r ∈ radical (m ⊔ span {x}) := fun x hxm =>
        by_contradiction fun hrmx => hxm <| by
          /-
            R : Type u
            inst✝ : CommSemiring R
            I : Ideal R
            r : R
            hr : Membership.mem (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime)) r
            hri : Not (Membership.mem I.radical r)
            m : Ideal R
            hIm : LE.le I m
            hm : Maximal (fun x => Membership.mem (setOf fun K => Not (Membership.mem K.ra …
            hrm : Not (Membership.mem m.radical r)
            x : R
            hxm : Not (Membership.mem m x)
            hrmx : Not (Membership.mem (Max.max m (Ideal.span (Singleton.singleton x))).ra …
            ⊢ Membership.mem m x
          -/
          rw [hm.eq_of_le hrmx le_sup_left]
          /-
            R : Type u
            inst✝ : CommSemiring R
            I : Ideal R
            r : R
            hr : Membership.mem (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime)) r
            hri : Not (Membership.mem I.radical r)
            m : Ideal R
            hIm : LE.le I m
            hm : Maximal (fun x => Membership.mem (setOf fun K => Not (Membership.mem K.ra …
            hrm : Not (Membership.mem m.radical r)
            x : R
            hxm : Not (Membership.mem m x)
            hrmx : Not (Membership.mem (Max.max m (Ideal.span (Singleton.singleton x))).ra …
            ⊢ Membership.mem (Max.max m (Ideal.span (Singleton.singleton x))) x
          -/
          exact Submodule.mem_sup_right <| mem_span_singleton_self x
          /-
            🎉 no goals
          -/
      have : IsPrime m :=
            /-
              R : Type u
              inst✝ : CommSemiring R
              I : Ideal R
              r : R
              hr : Membership.mem (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime)) r
              hri : Not (Membership.mem I.radical r)
              m : Ideal R
              hIm : LE.le I m
              hm : Maximal (fun x => Membership.mem (setOf fun K => Not (Membership.mem K.ra …
              hrm : Not (Membership.mem m.radical r)
              this : ∀ (x : R), Not (Membership.mem m x) → Membership.mem (Max.max m (Ideal. …
              ⊢ Ne m Top.top
            -/
        ⟨by rintro rfl; rw [radical_top] at hrm; exact hrm trivial, fun {x y} hxym =>
                                                 /-
                                                   🎉 no goals
                                                 -/
          or_iff_not_imp_left.2 fun hxm =>
            by_contradiction fun hym =>
              let ⟨n, hrn⟩ := this _ hxm
              let ⟨p, hpm, q, hq, hpqrn⟩ := Submodule.mem_sup.1 hrn
              let ⟨c, hcxq⟩ := mem_span_singleton'.1 hq
              let ⟨k, hrk⟩ := this _ hym
              let ⟨f, hfm, g, hg, hfgrk⟩ := Submodule.mem_sup.1 hrk
              let ⟨d, hdyg⟩ := mem_span_singleton'.1 hg
              hrm
                ⟨n + k, by
                  rw [pow_add, ← hpqrn, ← hcxq, ← hfgrk, ← hdyg, add_mul, mul_add (c * x),
                      mul_assoc c x (d * y), mul_left_comm x, ← mul_assoc]
                  refine
                    m.add_mem (m.mul_mem_right _ hpm)
                    (m.add_mem (m.mul_mem_left _ hfm) (m.mul_mem_left _ hxym))⟩⟩
    hrm <|
      this.radical.symm ▸ (sInf_le ⟨hIm, this⟩ : sInf { J : Ideal R | I ≤ J ∧ IsPrime J } ≤ m) hr


theorem isRadical_bot_of_noZeroDivisors {R} [CommSemiring R] [NoZeroDivisors R] :
    (⊥ : Ideal R).IsRadical := fun _ hx => hx.recOn fun _ hn => pow_eq_zero hn


@[simp]
theorem radical_bot_of_noZeroDivisors {R : Type u} [CommSemiring R] [NoZeroDivisors R] :
    radical (⊥ : Ideal R) = ⊥ :=
  eq_bot_iff.2 isRadical_bot_of_noZeroDivisors


instance : IdemCommSemiring (Ideal R) :=
  inferInstance


variable (R) in
theorem top_pow (n : ℕ) : (⊤ ^ n : Ideal R) = ⊤ :=
                                        /-
                                          R : Type u
                                          inst✝ : CommSemiring R
                                          n✝ n : Nat
                                          ih : Eq (HPow.hPow Top.top n) Top.top
                                          ⊢ Eq (HPow.hPow Top.top n.succ) Top.top
                                        -/
  Nat.recOn n one_eq_top fun n ih => by rw [pow_succ, ih, top_mul]
                                        /-
                                          🎉 no goals
                                        -/


lemma radical_pow : ∀ {n}, n ≠ 0 → radical (I ^ n) = radical I
               /-
                 R : Type u
                 inst✝ : CommSemiring R
                 I : Ideal R
                 x✝ : Ne 1 0
                 ⊢ Eq (HPow.hPow I 1).radical I.radical
               -/
  | 1, _ => by simp
               /-
                 🎉 no goals
               -/
                   /-
                     R : Type u
                     inst✝ : CommSemiring R
                     I : Ideal R
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 2) 0
                     ⊢ Eq (HPow.hPow I (HAdd.hAdd n 2)).radical I.radical
                   -/
  | n + 2, _ => by rw [pow_succ, radical_mul, radical_pow n.succ_ne_zero, inf_idem]
                   /-
                     🎉 no goals
                   -/


theorem IsPrime.mul_le {I J P : Ideal R} (hp : IsPrime P) : I * J ≤ P ↔ I ≤ P ∨ J ≤ P := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J P : Ideal R
    hp : P.IsPrime
    ⊢ Iff (LE.le (HMul.hMul I J) P) (Or (LE.le I P) (LE.le J P))
  -/
  rw [or_comm, Ideal.mul_le]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J P : Ideal R
    hp : P.IsPrime
    ⊢ Iff (∀ (r : R), Membership.mem I r → ∀ (s : R), Membership.mem J s → Members …
  -/
  simp_rw [hp.mul_mem_iff_mem_or_mem, SetLike.le_def, ← forall_or_left, or_comm, forall_or_left]
  /-
    🎉 no goals
  -/


theorem IsPrime.inf_le {I J P : Ideal R} (hp : IsPrime P) : I ⊓ J ≤ P ↔ I ≤ P ∨ J ≤ P :=
  ⟨fun h ↦ hp.mul_le.1 <| mul_le_inf.trans h, fun h ↦ h.elim inf_le_left.trans inf_le_right.trans⟩


theorem IsPrime.multiset_prod_le {s : Multiset (Ideal R)} {P : Ideal R} (hp : IsPrime P) :
    s.prod ≤ P ↔ ∃ I ∈ s, I ≤ P :=
                     /-
                       R : Type u
                       inst✝ : CommSemiring R
                       s : Multiset (Ideal R)
                       P : Ideal R
                       hp : P.IsPrime
                       ⊢ Iff (LE.le (Multiset.prod 0) P) (Exists fun I => And (Membership.mem 0 I) (L …
                     -/
                     /-
                       🎉 no goals
                     -/
  s.induction_on (by simp [hp.ne_top]) fun I s ih ↦ by simp [hp.mul_le, ih]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem IsPrime.multiset_prod_map_le {s : Multiset ι} (f : ι → Ideal R) {P : Ideal R}
    (hp : IsPrime P) : (s.map f).prod ≤ P ↔ ∃ i ∈ s, f i ≤ P := by
  /-
    R : Type u
    ι : Type u_1
    inst✝ : CommSemiring R
    s : Multiset ι
    f : ι → Ideal R
    P : Ideal R
    hp : P.IsPrime
    ⊢ Iff (LE.le (Multiset.map f s).prod P) (Exists fun i => And (Membership.mem s …
  -/
  simp_rw [hp.multiset_prod_le, Multiset.mem_map, exists_exists_and_eq_and]
  /-
    🎉 no goals
  -/


theorem IsPrime.multiset_prod_mem_iff_exists_mem {I : Ideal R} (hI : I.IsPrime) (s : Multiset R) :
    s.prod ∈ I ↔ ∃ p ∈ s, p ∈ I := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    s : Multiset R
    ⊢ Iff (Membership.mem I s.prod) (Exists fun p => And (Membership.mem s p) (Mem …
  -/
  simpa [span_singleton_le_iff_mem] using (hI.multiset_prod_map_le (span {·}))
  /-
    🎉 no goals
  -/


theorem IsPrime.pow_le_iff {I P : Ideal R} [hP : P.IsPrime] {n : ℕ} (hn : n ≠ 0) :
    I ^ n ≤ P ↔ I ≤ P := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I P : Ideal R
    hP : P.IsPrime
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LE.le (HPow.hPow I n) P) (LE.le I P)
  -/
  have h : (Multiset.replicate n I).prod ≤ P ↔ _ := hP.multiset_prod_le
  simp_rw [Multiset.prod_replicate, Multiset.mem_replicate, ne_eq, hn, not_false_eq_true,
    true_and, exists_eq_left] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    I P : Ideal R
    hP : P.IsPrime
    n : Nat
    hn : Ne n 0
    h : Iff (LE.le (HPow.hPow I n) P) (LE.le I P)
    ⊢ Iff (LE.le (HPow.hPow I n) P) (LE.le I P)
  -/
  exact h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-06")] alias pow_le_prime_iff := IsPrime.pow_le_iff


theorem IsPrime.le_of_pow_le {I P : Ideal R} [hP : P.IsPrime] {n : ℕ} (h : I ^ n ≤ P) :
    I ≤ P := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I P : Ideal R
    hP : P.IsPrime
    n : Nat
    h : LE.le (HPow.hPow I n) P
    ⊢ LE.le I P
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      I P : Ideal R
      hP : P.IsPrime
      n : Nat
      h : LE.le (HPow.hPow I n) P
      hn : Eq n 0
      ⊢ LE.le I P
    -/
  · rw [hn, pow_zero, one_eq_top] at h
    /-
      case pos
      R : Type u
      inst✝ : CommSemiring R
      I P : Ideal R
      hP : P.IsPrime
      n : Nat
      h : LE.le Top.top P
      hn : Eq n 0
      ⊢ LE.le I P
    -/
    exact fun ⦃_⦄ _ ↦ h Submodule.mem_top
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : CommSemiring R
      I P : Ideal R
      hP : P.IsPrime
      n : Nat
      h : LE.le (HPow.hPow I n) P
      hn : Not (Eq n 0)
      ⊢ LE.le I P
    -/
  · exact (pow_le_iff hn).mp h
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-06")] alias le_of_pow_le_prime := IsPrime.le_of_pow_le


theorem IsPrime.prod_le {s : Finset ι} {f : ι → Ideal R} {P : Ideal R} (hp : IsPrime P) :
    s.prod f ≤ P ↔ ∃ i ∈ s, f i ≤ P :=
  hp.multiset_prod_map_le f


@[deprecated (since := "2024-10-06")] alias prod_le_prime := IsPrime.prod_le


/-- The product of a finite number of elements in the commutative semiring `R` lies in the
  prime ideal `p` if and only if at least one of those elements is in `p`. -/
theorem IsPrime.prod_mem_iff {s : Finset ι} {x : ι → R} {p : Ideal R} [hp : p.IsPrime] :
    ∏ i in s, x i ∈ p ↔ ∃ i ∈ s, x i ∈ p := by
  /-
    R : Type u
    ι : Type u_1
    inst✝ : CommSemiring R
    s : Finset ι
    x : ι → R
    p : Ideal R
    hp : p.IsPrime
    ⊢ Iff (Membership.mem p (s.prod fun i => x i)) (Exists fun i => And (Membershi …
  -/
  simp_rw [← span_singleton_le_iff_mem, ← prod_span_singleton]
  /-
    R : Type u
    ι : Type u_1
    inst✝ : CommSemiring R
    s : Finset ι
    x : ι → R
    p : Ideal R
    hp : p.IsPrime
    ⊢ Iff (LE.le (s.prod fun i => Ideal.span (Singleton.singleton (x i))) p) (Exis …
  -/
  exact hp.prod_le
  /-
    🎉 no goals
  -/


theorem IsPrime.prod_mem_iff_exists_mem {I : Ideal R} (hI : I.IsPrime) (s : Finset R) :
    s.prod (fun x ↦ x) ∈ I ↔ ∃ p ∈ s, p ∈ I := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    s : Finset R
    ⊢ Iff (Membership.mem I (s.prod fun x => x)) (Exists fun p => And (Membership. …
  -/
  rw [Finset.prod_eq_multiset_prod, Multiset.map_id']
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    s : Finset R
    ⊢ Iff (Membership.mem I s.val.prod) (Exists fun p => And (Membership.mem s p)  …
  -/
  exact hI.multiset_prod_mem_iff_exists_mem s.val
  /-
    🎉 no goals
  -/


theorem IsPrime.inf_le' {s : Finset ι} {f : ι → Ideal R} {P : Ideal R} (hp : IsPrime P) :
    s.inf f ≤ P ↔ ∃ i ∈ s, f i ≤ P :=
  ⟨fun h ↦ hp.prod_le.1 <| prod_le_inf.trans h, fun ⟨_, his, hip⟩ ↦ (Finset.inf_le his).trans hip⟩

-- Porting note: needed to add explicit coercions (· : Set R).

theorem subset_union {R : Type u} [Ring R] {I J K : Ideal R} :
    (I : Set R) ⊆ J ∪ K ↔ I ≤ J ∨ I ≤ K :=
  AddSubgroupClass.subset_union


theorem subset_union_prime' {R : Type u} [CommRing R] {s : Finset ι} {f : ι → Ideal R} {a b : ι}
    (hp : ∀ i ∈ s, IsPrime (f i)) {I : Ideal R} :
    ((I : Set R) ⊆ f a ∪ f b ∪ ⋃ i ∈ (↑s : Set ι), f i) ↔ I ≤ f a ∨ I ≤ f b ∨ ∃ i ∈ s, I ≤ f i := by
  suffices
    ((I : Set R) ⊆ f a ∪ f b ∪ ⋃ i ∈ (↑s : Set ι), f i) → I ≤ f a ∨ I ≤ f b ∨ ∃ i ∈ s, I ≤ f i from
    ⟨this, fun h =>
      Or.casesOn h
        (fun h =>
          Set.Subset.trans h <|
            Set.Subset.trans Set.subset_union_left Set.subset_union_left)
        fun h =>
        Or.casesOn h
          (fun h =>
            Set.Subset.trans h <|
              Set.Subset.trans Set.subset_union_right Set.subset_union_left)
          fun ⟨i, his, hi⟩ => by
          refine Set.Subset.trans hi <| Set.Subset.trans ?_ Set.subset_union_right
          exact Set.subset_biUnion_of_mem (u := fun x ↦ (f x : Set R)) (Finset.mem_coe.2 his)⟩
  /-
    ι : Type u_1
    R : Type u
    inst✝ : CommRing R
    s : Finset ι
    f : ι → Ideal R
    a b : ι
    hp : ∀ (i : ι), Membership.mem s i → (f i).IsPrime
    I : Ideal R
    ⊢ HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion f …
  -/
  generalize hn : s.card = n; intro h
  /-
    ι : Type u_1
    R : Type u
    inst✝ : CommRing R
    s : Finset ι
    f : ι → Ideal R
    a b : ι
    hp : ∀ (i : ι), Membership.mem s i → (f i).IsPrime
    I : Ideal R
    n : Nat
    hn : Eq s.card n
    h : HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion …
    ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
  -/
  induction' n with n ih generalizing a b s
    /-
      case zero
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      f : ι → Ideal R
      I : Ideal R
      s : Finset ι
      a b : ι
      hp : ∀ (i : ι), Membership.mem s i → (f i).IsPrime
      hn : Eq s.card 0
      h : HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion …
      ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
    -/
  · clear hp
    /-
      case zero
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      f : ι → Ideal R
      I : Ideal R
      s : Finset ι
      a b : ι
      hn : Eq s.card 0
      h : HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion …
      ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
    -/
    rw [Finset.card_eq_zero] at hn
    /-
      case zero
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      f : ι → Ideal R
      I : Ideal R
      s : Finset ι
      a b : ι
      hn : Eq s EmptyCollection.emptyCollection
      h : HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion …
      ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
    -/
    subst hn
    /-
      case zero
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      f : ι → Ideal R
      I : Ideal R
      a b : ι
      h : HasSubset.Subset (↑I) (Union.union (Union.union ↑(f a) ↑(f b)) (Set.iUnion …
      ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
    -/
    rw [Finset.coe_empty, Set.biUnion_empty, Set.union_empty, subset_union] at h
    /-
      case zero
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      f : ι → Ideal R
      I : Ideal R
      a b : ι
      h : Or (LE.le I (f a)) (LE.le I (f b))
      ⊢ Or (LE.le I (f a)) (Or (LE.le I (f b)) (Exists fun i => And (Membership.mem  …
    -/
    simpa only [exists_prop, Finset.not_mem_empty, false_and, exists_false, or_false]
    /-
      🎉 no goals
    -/
  classical
    replace hn : ∃ (i : ι) (t : Finset ι), i ∉ t ∧ insert i t = s ∧ t.card = n :=
      Finset.card_eq_succ.1 hn
    rcases hn with ⟨i, t, hit, rfl, hn⟩
    replace hp : IsPrime (f i) ∧ ∀ x ∈ t, IsPrime (f x) := (t.forall_mem_insert _ _).1 hp
    by_cases Ht : ∃ j ∈ t, f j ≤ f i
    · obtain ⟨j, hjt, hfji⟩ : ∃ j ∈ t, f j ≤ f i := Ht
      obtain ⟨u, hju, rfl⟩ : ∃ u, j ∉ u ∧ insert j u = t :=
        ⟨t.erase j, t.not_mem_erase j, Finset.insert_erase hjt⟩
      have hp' : ∀ k ∈ insert i u, IsPrime (f k) := by
        rw [Finset.forall_mem_insert] at hp ⊢
        exact ⟨hp.1, hp.2.2⟩
      have hiu : i ∉ u := mt Finset.mem_insert_of_mem hit
      have hn' : (insert i u).card = n := by
        rwa [Finset.card_insert_of_not_mem] at hn ⊢
        exacts [hiu, hju]
      have h' : (I : Set R) ⊆ f a ∪ f b ∪ ⋃ k ∈ (↑(insert i u) : Set ι), f k := by
        rw [Finset.coe_insert] at h ⊢
        rw [Finset.coe_insert] at h
        simp only [Set.biUnion_insert] at h ⊢
        rw [← Set.union_assoc (f i : Set R),
            Set.union_eq_self_of_subset_right hfji] at h
        exact h
      specialize ih hp' hn' h'
      refine ih.imp id (Or.imp id (Exists.imp fun k => ?_))
      exact And.imp (fun hk => Finset.insert_subset_insert i (Finset.subset_insert j u) hk) id
    by_cases Ha : f a ≤ f i
    · have h' : (I : Set R) ⊆ f i ∪ f b ∪ ⋃ j ∈ (↑t : Set ι), f j := by
        rw [Finset.coe_insert, Set.biUnion_insert, ← Set.union_assoc,
          Set.union_right_comm (f a : Set R),
          Set.union_eq_self_of_subset_left Ha] at h
        exact h
      specialize ih hp.2 hn h'
      right
      rcases ih with (ih | ih | ⟨k, hkt, ih⟩)
      · exact Or.inr ⟨i, Finset.mem_insert_self i t, ih⟩
      · exact Or.inl ih
      · exact Or.inr ⟨k, Finset.mem_insert_of_mem hkt, ih⟩
    by_cases Hb : f b ≤ f i
    · have h' : (I : Set R) ⊆ f a ∪ f i ∪ ⋃ j ∈ (↑t : Set ι), f j := by
        rw [Finset.coe_insert, Set.biUnion_insert, ← Set.union_assoc,
          Set.union_assoc (f a : Set R),
          Set.union_eq_self_of_subset_left Hb] at h
        exact h
      specialize ih hp.2 hn h'
      rcases ih with (ih | ih | ⟨k, hkt, ih⟩)
      · exact Or.inl ih
      · exact Or.inr (Or.inr ⟨i, Finset.mem_insert_self i t, ih⟩)
      · exact Or.inr (Or.inr ⟨k, Finset.mem_insert_of_mem hkt, ih⟩)
    by_cases Hi : I ≤ f i
    · exact Or.inr (Or.inr ⟨i, Finset.mem_insert_self i t, Hi⟩)
    have : ¬I ⊓ f a ⊓ f b ⊓ t.inf f ≤ f i := by
      simp only [hp.1.inf_le, hp.1.inf_le', not_or]
      exact ⟨⟨⟨Hi, Ha⟩, Hb⟩, Ht⟩
    rcases Set.not_subset.1 this with ⟨r, ⟨⟨⟨hrI, hra⟩, hrb⟩, hr⟩, hri⟩
    by_cases HI : (I : Set R) ⊆ f a ∪ f b ∪ ⋃ j ∈ (↑t : Set ι), f j
    · specialize ih hp.2 hn HI
      rcases ih with (ih | ih | ⟨k, hkt, ih⟩)
      · left
        exact ih
      · right
        left
        exact ih
      · right
        right
        exact ⟨k, Finset.mem_insert_of_mem hkt, ih⟩
    exfalso
    rcases Set.not_subset.1 HI with ⟨s, hsI, hs⟩
    rw [Finset.coe_insert, Set.biUnion_insert] at h
    have hsi : s ∈ f i := ((h hsI).resolve_left (mt Or.inl hs)).resolve_right (mt Or.inr hs)
    rcases h (I.add_mem hrI hsI) with (⟨ha | hb⟩ | hi | ht)
    · exact hs (Or.inl <| Or.inl <| add_sub_cancel_left r s ▸ (f a).sub_mem ha hra)
    · exact hs (Or.inl <| Or.inr <| add_sub_cancel_left r s ▸ (f b).sub_mem hb hrb)
    · exact hri (add_sub_cancel_right r s ▸ (f i).sub_mem hi hsi)
    · rw [Set.mem_iUnion₂] at ht
      rcases ht with ⟨j, hjt, hj⟩
      simp only [Finset.inf_eq_iInf, SetLike.mem_coe, Submodule.mem_iInf] at hr
      exact hs <| Or.inr <| Set.mem_biUnion hjt <|
        add_sub_cancel_left r s ▸ (f j).sub_mem hj <| hr j hjt


/-- Prime avoidance. Atiyah-Macdonald 1.11, Eisenbud 3.3, Stacks 00DS, Matsumura Ex.1.6. -/
theorem subset_union_prime {R : Type u} [CommRing R] {s : Finset ι} {f : ι → Ideal R} (a b : ι)
    (hp : ∀ i ∈ s, i ≠ a → i ≠ b → IsPrime (f i)) {I : Ideal R} :
    ((I : Set R) ⊆ ⋃ i ∈ (↑s : Set ι), f i) ↔ ∃ i ∈ s, I ≤ f i :=
  suffices ((I : Set R) ⊆ ⋃ i ∈ (↑s : Set ι), f i) → ∃ i, i ∈ s ∧ I ≤ f i by
    /-
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      s : Finset ι
      f : ι → Ideal R
      a b : ι
      hp : ∀ (i : ι), Membership.mem s i → Ne i a → Ne i b → (f i).IsPrime
      I : Ideal R
      this : HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))  …
      ⊢ Iff (HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))) …
    -/
    have aux := fun h => (bex_def.2 <| this h)
    /-
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      s : Finset ι
      f : ι → Ideal R
      a b : ι
      hp : ∀ (i : ι), Membership.mem s i → Ne i a → Ne i b → (f i).IsPrime
      I : Ideal R
      this : HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))  …
      aux : HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i)) → …
      ⊢ Iff (HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))) …
    -/
    simp_rw [exists_prop] at aux
    /-
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      s : Finset ι
      f : ι → Ideal R
      a b : ι
      hp : ∀ (i : ι), Membership.mem s i → Ne i a → Ne i b → (f i).IsPrime
      I : Ideal R
      this aux : HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f  …
      ⊢ Iff (HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))) …
    -/
    refine ⟨aux, fun ⟨i, his, hi⟩ ↦ Set.Subset.trans hi ?_⟩
    /-
      ι : Type u_1
      R : Type u
      inst✝ : CommRing R
      s : Finset ι
      f : ι → Ideal R
      a b : ι
      hp : ∀ (i : ι), Membership.mem s i → Ne i a → Ne i b → (f i).IsPrime
      I : Ideal R
      this aux : HasSubset.Subset (↑I) (Set.iUnion fun i => Set.iUnion fun h => ↑(f  …
      x✝ : Exists fun i => And (Membership.mem s i) (LE.le I (f i))
      i : ι
      his : Membership.mem s i
      hi : LE.le I (f i)
      ⊢ HasSubset.Subset (↑(f i)) (Set.iUnion fun i => Set.iUnion fun h => ↑(f i))
    -/
    apply Set.subset_biUnion_of_mem (show i ∈ (↑s : Set ι) from his)
    /-
      🎉 no goals
    -/
  fun h : (I : Set R) ⊆ ⋃ i ∈ (↑s : Set ι), f i => by
  classical
    by_cases has : a ∈ s
    · obtain ⟨t, hat, rfl⟩ : ∃ t, a ∉ t ∧ insert a t = s :=
        ⟨s.erase a, Finset.not_mem_erase a s, Finset.insert_erase has⟩
      by_cases hbt : b ∈ t
      · obtain ⟨u, hbu, rfl⟩ : ∃ u, b ∉ u ∧ insert b u = t :=
          ⟨t.erase b, Finset.not_mem_erase b t, Finset.insert_erase hbt⟩
        have hp' : ∀ i ∈ u, IsPrime (f i) := by
          intro i hiu
          refine hp i (Finset.mem_insert_of_mem (Finset.mem_insert_of_mem hiu)) ?_ ?_ <;>
              rintro rfl <;>
            solve_by_elim only [Finset.mem_insert_of_mem, *]
        rw [Finset.coe_insert, Finset.coe_insert, Set.biUnion_insert, Set.biUnion_insert, ←
          Set.union_assoc, subset_union_prime' hp'] at h
        rwa [Finset.exists_mem_insert, Finset.exists_mem_insert]
      · have hp' : ∀ j ∈ t, IsPrime (f j) := by
          intro j hj
          refine hp j (Finset.mem_insert_of_mem hj) ?_ ?_ <;> rintro rfl <;>
            solve_by_elim only [Finset.mem_insert_of_mem, *]
        rw [Finset.coe_insert, Set.biUnion_insert, ← Set.union_self (f a : Set R),
          subset_union_prime' hp', ← or_assoc, or_self_iff] at h
        rwa [Finset.exists_mem_insert]
    · by_cases hbs : b ∈ s
      · obtain ⟨t, hbt, rfl⟩ : ∃ t, b ∉ t ∧ insert b t = s :=
          ⟨s.erase b, Finset.not_mem_erase b s, Finset.insert_erase hbs⟩
        have hp' : ∀ j ∈ t, IsPrime (f j) := by
          intro j hj
          refine hp j (Finset.mem_insert_of_mem hj) ?_ ?_ <;> rintro rfl <;>
            solve_by_elim only [Finset.mem_insert_of_mem, *]
        rw [Finset.coe_insert, Set.biUnion_insert, ← Set.union_self (f b : Set R),
          subset_union_prime' hp', ← or_assoc, or_self_iff] at h
        rwa [Finset.exists_mem_insert]
      rcases s.eq_empty_or_nonempty with hse | hsne
      · subst hse
        rw [Finset.coe_empty, Set.biUnion_empty, Set.subset_empty_iff] at h
        have : (I : Set R) ≠ ∅ := Set.Nonempty.ne_empty (Set.nonempty_of_mem I.zero_mem)
        exact absurd h this
      · cases' hsne with i his
        obtain ⟨t, _, rfl⟩ : ∃ t, i ∉ t ∧ insert i t = s :=
          ⟨s.erase i, Finset.not_mem_erase i s, Finset.insert_erase his⟩
        have hp' : ∀ j ∈ t, IsPrime (f j) := by
          intro j hj
          refine hp j (Finset.mem_insert_of_mem hj) ?_ ?_ <;> rintro rfl <;>
            solve_by_elim only [Finset.mem_insert_of_mem, *]
        rw [Finset.coe_insert, Set.biUnion_insert, ← Set.union_self (f i : Set R),
          subset_union_prime' hp', ← or_assoc, or_self_iff] at h
        rwa [Finset.exists_mem_insert]


/-- If `I` divides `J`, then `I` contains `J`.

In a Dedekind domain, to divide and contain are equivalent, see `Ideal.dvd_iff_le`.
-/
theorem le_of_dvd {I J : Ideal R} : I ∣ J → J ≤ I
  | ⟨_, h⟩ => h.symm ▸ le_trans mul_le_inf inf_le_left


@[simp]
theorem isUnit_iff {I : Ideal R} : IsUnit I ↔ I = ⊤ :=
  isUnit_iff_dvd_one.trans
    ((@one_eq_top R _).symm ▸
                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : CommSemiring R
                                                                      I : Ideal R
                                                                      h : Eq I Top.top
                                                                      ⊢ Eq Top.top (HMul.hMul I Top.top)
                                                                    -/
      ⟨fun h => eq_top_iff.mpr (Ideal.le_of_dvd h), fun h => ⟨⊤, by rw [mul_top, h]⟩⟩)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance uniqueUnits : Unique (Ideal R)ˣ where
  default := 1
                                                 /-
                                                   R : Type u
                                                   ι : Type u_1
                                                   inst✝ : CommSemiring R
                                                   I J K L : Ideal R
                                                   u : Units (Ideal R)
                                                   ⊢ Eq (↑u) 1
                                                 -/
  uniq u := Units.ext (show (u : Ideal R) = 1 by rw [isUnit_iff.mp u.isUnit, one_eq_top])
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A variant of `Finsupp.linearCombination` that takes in vectors valued in `I`. -/
noncomputable def finsuppTotal : (ι →₀ I) →ₗ[R] M :=
  (Finsupp.linearCombination R v).comp (Finsupp.mapRange.linearMap I.subtype)


theorem finsuppTotal_apply (f : ι →₀ I) :
    finsuppTotal ι M I v f = f.sum fun i x => (x : R) • v i := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    f : Finsupp ι (Subtype fun x => Membership.mem I x)
    ⊢ Eq ((Ideal.finsuppTotal ι M I v) f) (f.sum fun i x => HSMul.hSMul (↑x) (v i))
  -/
  dsimp [finsuppTotal]
  /-
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    f : Finsupp ι (Subtype fun x => Membership.mem I x)
    ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.mapRange Subtype.val ⋯ f)) (f.s …
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum_mapRange_index]
  /-
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    f : Finsupp ι (Subtype fun x => Membership.mem I x)
    ⊢ ∀ (a : ι), Eq (HSMul.hSMul 0 (v a)) 0
  -/
  exact fun _ => zero_smul _ _
  /-
    🎉 no goals
  -/


theorem finsuppTotal_apply_eq_of_fintype [Fintype ι] (f : ι →₀ I) :
    finsuppTotal ι M I v f = ∑ i, (f i : R) • v i := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    I : Ideal R
    v : ι → M
    inst✝ : Fintype ι
    f : Finsupp ι (Subtype fun x => Membership.mem I x)
    ⊢ Eq ((Ideal.finsuppTotal ι M I v) f) (Finset.univ.sum fun i => HSMul.hSMul (↑ …
  -/
  rw [finsuppTotal_apply, Finsupp.sum_fintype]
  /-
    case h
    ι : Type u_1
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    I : Ideal R
    v : ι → M
    inst✝ : Fintype ι
    f : Finsupp ι (Subtype fun x => Membership.mem I x)
    ⊢ ∀ (i : ι), Eq (HSMul.hSMul (↑0) (v i)) 0
  -/
  exact fun _ => zero_smul _ _
  /-
    🎉 no goals
  -/


theorem range_finsuppTotal :
    LinearMap.range (finsuppTotal ι M I v) = I • Submodule.span R (Set.range v) := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    ⊢ Eq (LinearMap.range (Ideal.finsuppTotal ι M I v)) (HSMul.hSMul I (Submodule. …
  -/
  ext
  /-
    case h
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    x✝ : M
    ⊢ Iff (Membership.mem (LinearMap.range (Ideal.finsuppTotal ι M I v)) x✝) (Memb …
  -/
  rw [Submodule.mem_ideal_smul_span_iff_exists_sum]
  /-
    case h
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    x✝ : M
    ⊢ Iff (Membership.mem (LinearMap.range (Ideal.finsuppTotal ι M I v)) x✝) (Exis …
  -/
  refine ⟨fun ⟨f, h⟩ => ⟨Finsupp.mapRange.linearMap I.subtype f, fun i => (f i).2, h⟩, ?_⟩
  /-
    case h
    ι : Type u_1
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    I : Ideal R
    v : ι → M
    x✝ : M
    ⊢ (Exists fun a => Exists fun x => Eq (a.sum fun i c => HSMul.hSMul c (v i)) x …
  -/
  rintro ⟨a, ha, rfl⟩
  classical
    refine ⟨a.mapRange (fun r => if h : r ∈ I then ⟨r, h⟩ else 0)
      (by simp only [Submodule.zero_mem, ↓reduceDIte]; rfl), ?_⟩
    rw [finsuppTotal_apply, Finsupp.sum_mapRange_index]
    · apply Finsupp.sum_congr
      intro i _
      rw [dif_pos (ha i)]
    · exact fun _ => zero_smul _ _


theorem Finsupp.mem_ideal_span_range_iff_exists_finsupp {x : R} {v : α → R} :
    x ∈ Ideal.span (Set.range v) ↔ ∃ c : α →₀ R, (c.sum fun i a => a * v i) = x :=
  Finsupp.mem_span_range_iff_exists_finsupp


/-- An element `x` lies in the span of `v` iff it can be written as sum `∑ cᵢ • vᵢ = x`.
-/
theorem mem_ideal_span_range_iff_exists_fun [Fintype α] {x : R} {v : α → R} :
    x ∈ Ideal.span (Set.range v) ↔ ∃ c : α → R, ∑ i, c i * v i = x :=
  mem_span_range_iff_exists_fun _


theorem Associates.mk_ne_zero' {R : Type*} [CommSemiring R] {r : R} :
    Associates.mk (Ideal.span {r} : Ideal R) ≠ 0 ↔ r ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r : R
    ⊢ Iff (Ne (Associates.mk (Ideal.span (Singleton.singleton r))) 0) (Ne r 0)
  -/
  rw [Associates.mk_ne_zero, Ideal.zero_eq_bot, Ne, Ideal.span_singleton_eq_bot]
  /-
    🎉 no goals
  -/


open scoped nonZeroDivisors in
theorem Ideal.span_singleton_nonZeroDivisors {R : Type*} [CommSemiring R] [NoZeroDivisors R]
    {r : R} : span {r} ∈ (Ideal R)⁰ ↔ r ∈ R⁰ := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    r : R
    ⊢ Iff (Membership.mem (nonZeroDivisors (Ideal R)) (Ideal.span (Singleton.singl …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      r : R
      h✝ : Subsingleton R
      ⊢ Iff (Membership.mem (nonZeroDivisors (Ideal R)) (Ideal.span (Singleton.singl …
    -/
  · exact ⟨fun _ _ _ ↦ Subsingleton.eq_zero _, fun _ _ _ ↦ Subsingleton.eq_zero _⟩
    /-
      🎉 no goals
    -/
  · rw [mem_nonZeroDivisors_iff_ne_zero, mem_nonZeroDivisors_iff_ne_zero, ne_eq, zero_eq_bot,
      span_singleton_eq_bot]


instance moduleSubmodule : Module (Ideal R) (Submodule R M) where
  smul_add := smul_sup
  add_smul := sup_smul
  mul_smul := Submodule.smul_assoc
                 /-
                   R : Type u
                   M : Type v
                   inst✝² : CommSemiring R
                   inst✝¹ : AddCommMonoid M
                   inst✝ : Module R M
                   ⊢ ∀ (b : Submodule R M), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by simp
                 /-
                   🎉 no goals
                 -/
  zero_smul := bot_smul
  smul_zero := smul_bot


lemma span_smul_eq
    (s : Set R) (N : Submodule R M) :
    Ideal.span s • N = s • N := by
  /-
    R : Type u
    M : Type v
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    N : Submodule R M
    ⊢ Eq (HSMul.hSMul (Ideal.span s) N) (HSMul.hSMul s N)
  -/
  rw [← coe_set_smul, coe_span_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem set_smul_top_eq_span (s : Set R) :
    s • ⊤ = Ideal.span s :=
  (span_smul_eq s ⊤).symm.trans (Ideal.span s).mul_top


instance algebraIdeal : Algebra (Ideal R) (Submodule R A) where
  __ := moduleSubmodule
  toFun := map (Algebra.linearMap R A)
  map_one' := by
    /-
      R : Type u
      M : Type v
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      A : Type ?u.386303
      B : Type ?u.386306
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      ⊢ Eq (Submodule.map (Algebra.linearMap R A) 1) 1
    -/
    rw [one_eq_span, map_span, Set.image_singleton, Algebra.linearMap_apply, map_one, one_eq_span]
    /-
      🎉 no goals
    -/
  map_mul' := (Submodule.map_mul · · <| Algebra.ofId R A)
  map_zero' := map_bot _
  map_add' := (map_sup · · _)
                                             /-
                                               R : Type u
                                               M✝ : Type v
                                               inst✝⁶ : CommSemiring R
                                               inst✝⁵ : AddCommMonoid M✝
                                               inst✝⁴ : Module R M✝
                                               A : Type ?u.386303
                                               B : Type ?u.386306
                                               inst✝³ : Semiring A
                                               inst✝² : Semiring B
                                               inst✝¹ : Algebra R A
                                               inst✝ : Algebra R B
                                               I : Ideal R
                                               M : Submodule R A
                                               ⊢ ∀ (m : A), Membership.mem ({ toFun := Submodule.map (Algebra.linearMap R A), …
                                             -/
  commutes' I M := mul_comm_of_commute <| by rintro _ ⟨r, _, rfl⟩ a _; apply Algebra.commutes
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  smul_def' I M := le_antisymm (smul_le.mpr fun r hr a ha ↦ by
    /-
      R : Type u
      M✝ : Type v
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M✝
      inst✝⁴ : Module R M✝
      A : Type ?u.386303
      B : Type ?u.386306
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal R
      M : Submodule R A
      r : R
      hr : Membership.mem I r
      a : A
      ha : Membership.mem M a
      ⊢ Membership.mem (HMul.hMul ({ toFun := Submodule.map (Algebra.linearMap R A), …
    -/
    rw [Algebra.smul_def]; exact Submodule.mul_mem_mul ⟨r, hr, rfl⟩ ha) (Submodule.mul_le.mpr <| by
                           /-
                             🎉 no goals
                           -/
    /-
      R : Type u
      M✝ : Type v
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M✝
      inst✝⁴ : Module R M✝
      A : Type ?u.386303
      B : Type ?u.386306
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal R
      M : Submodule R A
      ⊢ ∀ (m : A), Membership.mem ({ toFun := Submodule.map (Algebra.linearMap R A), …
    -/
    rintro _ ⟨r, hr, rfl⟩ a ha; rw [Algebra.linearMap_apply, ← Algebra.smul_def]
    /-
      case intro.intro
      R : Type u
      M✝ : Type v
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid M✝
      inst✝⁴ : Module R M✝
      A : Type ?u.386303
      B : Type ?u.386306
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal R
      M : Submodule R A
      r : R
      hr : Membership.mem (↑I) r
      a : A
      ha : Membership.mem M a
      ⊢ Membership.mem (HSMul.hSMul I M) (HSMul.hSMul r a)
    -/
    exact Submodule.smul_mem_smul hr ha)
    /-
      🎉 no goals
    -/


/-- `Submonoid.map` as an `AlgHom`, when applied to an `AlgHom`. -/
@[simps!] def mapAlgHom (f : A →ₐ[R] B) : Submodule R A →ₐ[Ideal R] Submodule R B where
  __ := mapHom f
  commutes' I := (map_comp _ _ I).symm.trans (congr_arg (map · I) <| LinearMap.ext f.commutes)


/-- `Submonoid.map` as an `AlgEquiv`, when applied to an `AlgEquiv`. -/
-- TODO: when A, B noncommutative, still has `MulEquiv`.
@[simps!] def mapAlgEquiv (f : A ≃ₐ[R] B) : Submodule R A ≃ₐ[Ideal R] Submodule R B where
  __ := mapAlgHom f
  invFun := mapAlgHom f.symm
  left_inv I := (map_comp _ _ I).symm.trans <|
    (congr_arg (map · I) <| LinearMap.ext (f.left_inv ·)).trans (map_id I)
  right_inv I := (map_comp _ _ I).symm.trans <|
    (congr_arg (map · I) <| LinearMap.ext (f.right_inv ·)).trans (map_id I)


instance {R} [Semiring R] : NonUnitalSubsemiringClass (Ideal R) R where
  mul_mem _ hb := Ideal.mul_mem_left _ _ hb

instance {R} [Ring R] : NonUnitalSubringClass (Ideal R) R where

