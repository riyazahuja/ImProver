theorem fg_bot : (⊥ : Submodule R M).FG :=
         /-
           R : Type u_1
           M : Type u_2
           inst✝² : Semiring R
           inst✝¹ : AddCommMonoid M
           inst✝ : Module R M
           ⊢ Eq (Submodule.span R ↑EmptyCollection.emptyCollection) Bot.bot
         -/
  ⟨∅, by rw [Finset.coe_empty, span_empty]⟩
         /-
           🎉 no goals
         -/


theorem fg_span {s : Set M} (hs : s.Finite) : FG (span R s) :=
                   /-
                     R : Type u_1
                     M : Type u_2
                     inst✝² : Semiring R
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     s : Set M
                     hs : s.Finite
                     ⊢ Eq (Submodule.span R ↑hs.toFinset) (Submodule.span R s)
                   -/
  ⟨hs.toFinset, by rw [hs.coe_toFinset]⟩
                   /-
                     🎉 no goals
                   -/


theorem fg_span_singleton (x : M) : FG (R ∙ x) :=
  fg_span (finite_singleton x)


theorem FG.sup {N₁ N₂ : Submodule R M} (hN₁ : N₁.FG) (hN₂ : N₂.FG) : (N₁ ⊔ N₂).FG :=
  let ⟨t₁, ht₁⟩ := fg_def.1 hN₁
  let ⟨t₂, ht₂⟩ := fg_def.1 hN₂
                                           /-
                                             R : Type u_1
                                             M : Type u_2
                                             inst✝² : Semiring R
                                             inst✝¹ : AddCommMonoid M
                                             inst✝ : Module R M
                                             N₁ N₂ : Submodule R M
                                             hN₁ : N₁.FG
                                             hN₂ : N₂.FG
                                             t₁ : Set M
                                             ht₁ : And t₁.Finite (Eq (Submodule.span R t₁) N₁)
                                             t₂ : Set M
                                             ht₂ : And t₂.Finite (Eq (Submodule.span R t₂) N₂)
                                             ⊢ Eq (Submodule.span R (Union.union t₁ t₂)) (Max.max N₁ N₂)
                                           -/
  fg_def.2 ⟨t₁ ∪ t₂, ht₁.1.union ht₂.1, by rw [span_union, ht₁.2, ht₂.2]⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem fg_finset_sup {ι : Type*} (s : Finset ι) (N : ι → Submodule R M) (h : ∀ i ∈ s, (N i).FG) :
    (s.sup N).FG :=
  Finset.sup_induction fg_bot (fun _ ha _ hb => ha.sup hb) h


theorem fg_biSup {ι : Type*} (s : Finset ι) (N : ι → Submodule R M) (h : ∀ i ∈ s, (N i).FG) :
                            /-
                              R : Type u_1
                              M : Type u_2
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              ι : Type u_3
                              s : Finset ι
                              N : ι → Submodule R M
                              h : ∀ (i : ι), Membership.mem s i → (N i).FG
                              ⊢ (iSup fun i => iSup fun h => N i).FG
                            -/
    (⨆ i ∈ s, N i).FG := by simpa only [Finset.sup_eq_iSup] using fg_finset_sup s N h
                            /-
                              🎉 no goals
                            -/


theorem fg_iSup {ι : Sort*} [Finite ι] (N : ι → Submodule R M) (h : ∀ i, (N i).FG) :
    (iSup N).FG := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Sort u_3
    inst✝ : Finite ι
    N : ι → Submodule R M
    h : ∀ (i : ι), (N i).FG
    ⊢ (iSup N).FG
  -/
  cases nonempty_fintype (PLift ι)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Sort u_3
    inst✝ : Finite ι
    N : ι → Submodule R M
    h : ∀ (i : ι), (N i).FG
    val✝ : Fintype (PLift ι)
    ⊢ (iSup N).FG
  -/
  simpa [iSup_plift_down] using fg_biSup Finset.univ (N ∘ PLift.down) fun i _ => h i.down
  /-
    🎉 no goals
  -/


theorem fg_pi {ι : Type*} {M : ι → Type*} [Finite ι] [∀ i, AddCommMonoid (M i)]
    [∀ i, Module R (M i)] {p : ∀ i, Submodule R (M i)} (hsb : ∀ i, (p i).FG) :
    (Submodule.pi Set.univ p).FG := by
  classical
    simp_rw [fg_def] at hsb ⊢
    choose t htf hts using hsb
    refine
      ⟨⋃ i, (LinearMap.single R _ i) '' t i, Set.finite_iUnion fun i => (htf i).image _, ?_⟩
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 changed `span_image` into `span_image _`
    simp_rw [span_iUnion, span_image _, hts, Submodule.iSup_map_single]


theorem FG.map {N : Submodule R M} (hs : N.FG) : (N.map f).FG :=
  let ⟨t, ht⟩ := fg_def.1 hs
                                     /-
                                       R : Type u_1
                                       M : Type u_2
                                       inst✝⁴ : Semiring R
                                       inst✝³ : AddCommMonoid M
                                       inst✝² : Module R M
                                       P : Type u_3
                                       inst✝¹ : AddCommMonoid P
                                       inst✝ : Module R P
                                       f : LinearMap (RingHom.id R) M P
                                       N : Submodule R M
                                       hs : N.FG
                                       t : Set M
                                       ht : And t.Finite (Eq (Submodule.span R t) N)
                                       ⊢ Eq (Submodule.span R (Set.image (⇑f) t)) (Submodule.map f N)
                                     -/
  fg_def.2 ⟨f '' t, ht.1.image _, by rw [span_image, ht.2]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem fg_of_fg_map_injective (f : M →ₗ[R] P) (hf : Function.Injective f) {N : Submodule R M}
    (hfn : (N.map f).FG) : N.FG :=
  let ⟨t, ht⟩ := hfn
  ⟨t.preimage f fun _ _ _ _ h => hf h,
    Submodule.map_injective_of_injective hf <| by
      rw [map_span, Finset.coe_preimage, Set.image_preimage_eq_inter_range,
        Set.inter_eq_self_of_subset_left, ht]
      /-
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        P : Type u_3
        inst✝¹ : AddCommMonoid P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        hf : Function.Injective ⇑f
        N : Submodule R M
        hfn : (Submodule.map f N).FG
        t : Finset P
        ht : Eq (Submodule.span R ↑t) (Submodule.map f N)
        ⊢ HasSubset.Subset (↑t) (Set.range ⇑f)
      -/
      rw [← LinearMap.range_coe, ← span_le, ht, ← map_top]
      /-
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        P : Type u_3
        inst✝¹ : AddCommMonoid P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        hf : Function.Injective ⇑f
        N : Submodule R M
        hfn : (Submodule.map f N).FG
        t : Finset P
        ht : Eq (Submodule.span R ↑t) (Submodule.map f N)
        ⊢ LE.le (Submodule.map f N) (Submodule.map f Top.top)
      -/
      exact map_mono le_top⟩
      /-
        🎉 no goals
      -/


theorem fg_of_fg_map {R M P : Type*} [Ring R] [AddCommGroup M] [Module R M] [AddCommGroup P]
    [Module R P] (f : M →ₗ[R] P)
    (hf : LinearMap.ker f = ⊥) {N : Submodule R M}
    (hfn : (N.map f).FG) : N.FG :=
  fg_of_fg_map_injective f (LinearMap.ker_eq_bot.1 hf) hfn


theorem fg_top (N : Submodule R M) : (⊤ : Submodule R N).FG ↔ N.FG :=
  ⟨fun h => N.range_subtype ▸ map_top N.subtype ▸ h.map _, fun h =>
                                                                 /-
                                                                   R : Type u_1
                                                                   M : Type u_2
                                                                   inst✝² : Semiring R
                                                                   inst✝¹ : AddCommMonoid M
                                                                   inst✝ : Module R M
                                                                   N : Submodule R M
                                                                   h : N.FG
                                                                   ⊢ (Submodule.map N.subtype Top.top).FG
                                                                 -/
    fg_of_fg_map_injective N.subtype Subtype.val_injective <| by rwa [map_top, range_subtype]⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem fg_of_linearEquiv (e : M ≃ₗ[R] P) (h : (⊤ : Submodule R P).FG) : (⊤ : Submodule R M).FG :=
  e.symm.range ▸ map_top (e.symm : P →ₗ[R] M) ▸ h.map _


theorem fg_induction (R M : Type*) [Semiring R] [AddCommMonoid M] [Module R M]
    (P : Submodule R M → Prop) (h₁ : ∀ x, P (Submodule.span R {x}))
    (h₂ : ∀ M₁ M₂, P M₁ → P M₂ → P (M₁ ⊔ M₂)) (N : Submodule R M) (hN : N.FG) : P N := by
  classical
    obtain ⟨s, rfl⟩ := hN
    induction s using Finset.induction with
    | empty =>
      rw [Finset.coe_empty, Submodule.span_empty, ← Submodule.span_zero_singleton]
      exact h₁ _
    | insert _ ih =>
      rw [Finset.coe_insert, Submodule.span_insert]
      exact h₂ _ _ (h₁ _) ih


theorem fg_restrictScalars {R S M : Type*} [CommSemiring R] [Semiring S] [Algebra R S]
    [AddCommGroup M] [Module S M] [Module R M] [IsScalarTower R S M] (N : Submodule S M)
    (hfin : N.FG) (h : Function.Surjective (algebraMap R S)) :
    (Submodule.restrictScalars R N).FG := by
  /-
    R : Type u_4
    S : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module S M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R S M
    N : Submodule S M
    hfin : N.FG
    h : Function.Surjective ⇑(algebraMap R S)
    ⊢ (Submodule.restrictScalars R N).FG
  -/
  obtain ⟨X, rfl⟩ := hfin
  /-
    case intro
    R : Type u_4
    S : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module S M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R S M
    h : Function.Surjective ⇑(algebraMap R S)
    X : Finset M
    ⊢ (Submodule.restrictScalars R (Submodule.span S ↑X)).FG
  -/
  use X
  /-
    case h
    R : Type u_4
    S : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommGroup M
    inst✝² : Module S M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R S M
    h : Function.Surjective ⇑(algebraMap R S)
    X : Finset M
    ⊢ Eq (Submodule.span R ↑X) (Submodule.restrictScalars R (Submodule.span S ↑X))
  -/
  exact (Submodule.restrictScalars_span R S h (X : Set M)).symm
  /-
    🎉 no goals
  -/


lemma FG.of_restrictScalars (R) {A M} [CommSemiring R] [Semiring A] [AddCommMonoid M]
    [Algebra R A] [Module R M] [Module A M] [IsScalarTower R A M] (S : Submodule A M)
    (hS : (S.restrictScalars R).FG) : S.FG := by
  /-
    R : Type u_4
    A : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Algebra R A
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    S : Submodule A M
    hS : (Submodule.restrictScalars R S).FG
    ⊢ S.FG
  -/
  obtain ⟨s, e⟩ := hS
  /-
    case intro
    R : Type u_4
    A : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Algebra R A
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    S : Submodule A M
    s : Finset M
    e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
    ⊢ S.FG
  -/
  refine ⟨s, Submodule.restrictScalars_injective R _ _ (le_antisymm ?_ ?_)⟩
    /-
      case intro.refine_1
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Algebra R A
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      S : Submodule A M
      s : Finset M
      e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
      ⊢ LE.le (Submodule.restrictScalars R (Submodule.span A ↑s)) (Submodule.restric …
    -/
  · show Submodule.span A s ≤ S
    /-
      case intro.refine_1
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Algebra R A
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      S : Submodule A M
      s : Finset M
      e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
      ⊢ LE.le (Submodule.span A ↑s) S
    -/
    have := Submodule.span_le.mp e.le
    /-
      case intro.refine_1
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Algebra R A
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      S : Submodule A M
      s : Finset M
      e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
      this : HasSubset.Subset ↑s ↑(Submodule.restrictScalars R S)
      ⊢ LE.le (Submodule.span A ↑s) S
    -/
    rwa [Submodule.span_le]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Algebra R A
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      S : Submodule A M
      s : Finset M
      e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
      ⊢ LE.le (Submodule.restrictScalars R S) (Submodule.restrictScalars R (Submodul …
    -/
  · rw [← e]
    /-
      case intro.refine_2
      R : Type u_4
      A : Type u_5
      M : Type u_6
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Algebra R A
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      S : Submodule A M
      s : Finset M
      e : Eq (Submodule.span R ↑s) (Submodule.restrictScalars R S)
      ⊢ LE.le (Submodule.span R ↑s) (Submodule.restrictScalars R (Submodule.span A ↑ …
    -/
    exact Submodule.span_le_restrictScalars _ _ _
    /-
      🎉 no goals
    -/


theorem FG.stabilizes_of_iSup_eq {M' : Submodule R M} (hM' : M'.FG) (N : ℕ →o Submodule R M)
    (H : iSup N = M') : ∃ n, M' = N n := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    hM' : M'.FG
    N : OrderHom Nat (Submodule R M)
    H : Eq (iSup ⇑N) M'
    ⊢ Exists fun n => Eq M' (N n)
  -/
  obtain ⟨S, hS⟩ := hM'
  have : ∀ s : S, ∃ n, (s : M) ∈ N n := fun s =>
    (Submodule.mem_iSup_of_chain N s).mp
      (by
        rw [H, ← hS]
        exact Submodule.subset_span s.2)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    N : OrderHom Nat (Submodule R M)
    H : Eq (iSup ⇑N) M'
    S : Finset M
    hS : Eq (Submodule.span R ↑S) M'
    this : ∀ (s : Subtype fun x => Membership.mem S x), Exists fun n => Membership …
    ⊢ Exists fun n => Eq M' (N n)
  -/
  choose f hf using this
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    N : OrderHom Nat (Submodule R M)
    H : Eq (iSup ⇑N) M'
    S : Finset M
    hS : Eq (Submodule.span R ↑S) M'
    f : (Subtype fun x => Membership.mem S x) → Nat
    hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
    ⊢ Exists fun n => Eq M' (N n)
  -/
  use S.attach.sup f
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    M' : Submodule R M
    N : OrderHom Nat (Submodule R M)
    H : Eq (iSup ⇑N) M'
    S : Finset M
    hS : Eq (Submodule.span R ↑S) M'
    f : (Subtype fun x => Membership.mem S x) → Nat
    hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
    ⊢ Eq M' (N (S.attach.sup f))
  -/
  apply le_antisymm
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      ⊢ LE.le M' (N (S.attach.sup f))
    -/
  · conv_lhs => rw [← hS]
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      ⊢ LE.le (Submodule.span R ↑S) (N (S.attach.sup f))
    -/
    rw [Submodule.span_le]
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      ⊢ HasSubset.Subset ↑S ↑(N (S.attach.sup f))
    -/
    intro s hs
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      s : M
      hs : Membership.mem (↑S) s
      ⊢ Membership.mem (↑(N (S.attach.sup f))) s
    -/
    exact N.2 (Finset.le_sup <| S.mem_attach ⟨s, hs⟩) (hf _)
    /-
      🎉 no goals
    -/
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      ⊢ LE.le (N (S.attach.sup f)) M'
    -/
  · rw [← H]
    /-
      case h.a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      M' : Submodule R M
      N : OrderHom Nat (Submodule R M)
      H : Eq (iSup ⇑N) M'
      S : Finset M
      hS : Eq (Submodule.span R ↑S) M'
      f : (Subtype fun x => Membership.mem S x) → Nat
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (N (f s)) ↑s
      ⊢ LE.le (N (S.attach.sup f)) (iSup ⇑N)
    -/
    exact le_iSup _ _
    /-
      🎉 no goals
    -/


/-- Finitely generated submodules are precisely compact elements in the submodule lattice. -/
theorem fg_iff_compact (s : Submodule R M) : s.FG ↔ CompleteLattice.IsCompactElement s := by
  classical
    -- Introduce shorthand for span of an element
    let sp : M → Submodule R M := fun a => span R {a}
    -- Trivial rewrite lemma; a small hack since simp (only) & rw can't accomplish this smoothly.
    have supr_rw : ∀ t : Finset M, ⨆ x ∈ t, sp x = ⨆ x ∈ (↑t : Set M), sp x := fun t => by rfl
    constructor
    · rintro ⟨t, rfl⟩
      rw [span_eq_iSup_of_singleton_spans, ← supr_rw, ← Finset.sup_eq_iSup t sp]
      apply CompleteLattice.isCompactElement_finsetSup
      exact fun n _ => singleton_span_isCompactElement n
    · intro h
      -- s is the Sup of the spans of its elements.
      have sSup' : s = sSup (sp '' ↑s) := by
        rw [sSup_eq_iSup, iSup_image, ← span_eq_iSup_of_singleton_spans, eq_comm, span_eq]
      -- by h, s is then below (and equal to) the sup of the spans of finitely many elements.
      obtain ⟨u, ⟨huspan, husup⟩⟩ := h (sp '' ↑s) (le_of_eq sSup')
      have ssup : s = u.sup id := by
        suffices u.sup id ≤ s from le_antisymm husup this
        rw [sSup', Finset.sup_id_eq_sSup]
        exact sSup_le_sSup huspan
      obtain ⟨t, -, rfl⟩ := Finset.subset_set_image_iff.mp huspan
      rw [Finset.sup_image, Function.id_comp, Finset.sup_eq_iSup, supr_rw, ←
        span_eq_iSup_of_singleton_spans, eq_comm] at ssup
      exact ⟨t, ssup⟩


instance (priority := 100) of_finite [Finite M] : Module.Finite R M := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    N : Type u_5
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : Finite M
    ⊢ Module.Finite R M
  -/
  cases nonempty_fintype M
  /-
    case intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    M : Type u_4
    N : Type u_5
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : Finite M
    val✝ : Fintype M
    ⊢ Module.Finite R M
  -/
  exact ⟨⟨Finset.univ, by rw [Finset.coe_univ]; exact Submodule.span_univ⟩⟩
  /-
    🎉 no goals
  -/


theorem of_surjective [hM : Module.Finite R M] (f : M →ₗ[R] N) (hf : Surjective f) :
    Module.Finite R N :=
  ⟨by
    /-
      R : Type u_1
      M : Type u_4
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      hM : Module.Finite R M
      f : LinearMap (RingHom.id R) M N
      hf : Function.Surjective ⇑f
      ⊢ Top.top.FG
    -/
    rw [← LinearMap.range_eq_top.2 hf, ← Submodule.map_top]
    /-
      R : Type u_1
      M : Type u_4
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      hM : Module.Finite R M
      f : LinearMap (RingHom.id R) M N
      hf : Function.Surjective ⇑f
      ⊢ (Submodule.map f Top.top).FG
    -/
    exact hM.1.map f⟩
    /-
      🎉 no goals
    -/


instance quotient (R) {A M} [Semiring R] [AddCommGroup M] [Ring A] [Module A M] [Module R M]
    [SMul R A] [IsScalarTower R A M] [Module.Finite R M]
    (N : Submodule A M) : Module.Finite R (M ⧸ N) :=
  Module.Finite.of_surjective (N.mkQ.restrictScalars R) N.mkQ_surjective


/-- The range of a linear map from a finite module is finite. -/
instance range {F : Type*} [FunLike F M N] [SemilinearMapClass F (RingHom.id R) M N]
    [Module.Finite R M] (f : F) : Module.Finite R (LinearMap.range f) :=
  of_surjective (SemilinearMapClass.semilinearMap f).rangeRestrict
    fun ⟨_, y, hy⟩ => ⟨y, Subtype.ext hy⟩


/-- Pushforwards of finite submodules are finite. -/
instance map (p : Submodule R M) [Module.Finite R p] (f : M →ₗ[R] N) : Module.Finite R (p.map f) :=
  of_surjective (f.restrict fun _ => Submodule.mem_map_of_mem) fun ⟨_, _, hy, hy'⟩ =>
    ⟨⟨_, hy⟩, Subtype.ext hy'⟩


instance pi {ι : Type*} {M : ι → Type*} [_root_.Finite ι] [∀ i, AddCommMonoid (M i)]
    [∀ i, Module R (M i)] [h : ∀ i, Module.Finite R (M i)] : Module.Finite R (∀ i, M i) :=
  ⟨by
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      M✝ : Type u_4
      N : Type u_5
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M✝
      inst✝⁵ : Module R M✝
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R N
      ι : Type u_6
      M : ι → Type u_7
      inst✝² : Finite ι
      inst✝¹ : (i : ι) → AddCommMonoid (M i)
      inst✝ : (i : ι) → Module R (M i)
      h : ∀ (i : ι), Module.Finite R (M i)
      ⊢ Top.top.FG
    -/
    rw [← Submodule.pi_top]
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      M✝ : Type u_4
      N : Type u_5
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M✝
      inst✝⁵ : Module R M✝
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R N
      ι : Type u_6
      M : ι → Type u_7
      inst✝² : Finite ι
      inst✝¹ : (i : ι) → AddCommMonoid (M i)
      inst✝ : (i : ι) → Module R (M i)
      h : ∀ (i : ι), Module.Finite R (M i)
      ⊢ (Submodule.pi ?m.61327 fun i => Top.top).FG
    -/
    exact Submodule.fg_pi fun i => (h i).1⟩
    /-
      🎉 no goals
    -/


instance self : Module.Finite R R :=
            /-
              R : Type u_1
              A : Type u_2
              B : Type u_3
              M : Type u_4
              N : Type u_5
              inst✝⁴ : Semiring R
              inst✝³ : AddCommMonoid M
              inst✝² : Module R M
              inst✝¹ : AddCommMonoid N
              inst✝ : Module R N
              ⊢ Eq (Submodule.span R ↑(Singleton.singleton 1)) Top.top
            -/
  ⟨⟨{1}, by simpa only [Finset.coe_singleton] using Ideal.span_singleton_one⟩⟩
            /-
              🎉 no goals
            -/


theorem of_restrictScalars_finite (R A M : Type*) [CommSemiring R] [Semiring A] [AddCommMonoid M]
    [Module R M] [Module A M] [Algebra R A] [IsScalarTower R A M] [hM : Module.Finite R M] :
    Module.Finite A M := by
  /-
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    hM : Module.Finite R M
    ⊢ Module.Finite A M
  -/
  rw [finite_def, Submodule.fg_def] at hM ⊢
  /-
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    hM : Exists fun S => And S.Finite (Eq (Submodule.span R S) Top.top)
    ⊢ Exists fun S => And S.Finite (Eq (Submodule.span A S) Top.top)
  -/
  obtain ⟨S, hSfin, hSgen⟩ := hM
  /-
    case intro.intro
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    S : Set M
    hSfin : S.Finite
    hSgen : Eq (Submodule.span R S) Top.top
    ⊢ Exists fun S => And S.Finite (Eq (Submodule.span A S) Top.top)
  -/
  refine ⟨S, hSfin, eq_top_iff.2 ?_⟩
  /-
    case intro.intro
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    S : Set M
    hSfin : S.Finite
    hSgen : Eq (Submodule.span R S) Top.top
    ⊢ LE.le Top.top (Submodule.span A S)
  -/
  have := Submodule.span_le_restrictScalars R A S
  /-
    case intro.intro
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    S : Set M
    hSfin : S.Finite
    hSgen : Eq (Submodule.span R S) Top.top
    this : LE.le (Submodule.span R S) (Submodule.restrictScalars R (Submodule.span …
    ⊢ LE.le Top.top (Submodule.span A S)
  -/
  rw [hSgen] at this
  /-
    case intro.intro
    R : Type u_6
    A : Type u_7
    M : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : Module A M
    inst✝¹ : Algebra R A
    inst✝ : IsScalarTower R A M
    S : Set M
    hSfin : S.Finite
    hSgen : Eq (Submodule.span R S) Top.top
    this : LE.le Top.top (Submodule.restrictScalars R (Submodule.span A S))
    ⊢ LE.le Top.top (Submodule.span A S)
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem equiv [Module.Finite R M] (e : M ≃ₗ[R] N) : Module.Finite R N :=
  of_surjective (e : M →ₗ[R] N) e.surjective


theorem equiv_iff (e : M ≃ₗ[R] N) : Module.Finite R M ↔ Module.Finite R N :=
  ⟨fun _ ↦ equiv e, fun _ ↦ equiv e.symm⟩


instance ulift [Module.Finite R M] : Module.Finite R (ULift M) := equiv ULift.moduleEquiv.symm


theorem iff_fg {N : Submodule R M} : Module.Finite R N ↔ N.FG := Module.finite_def.trans (fg_top _)


instance bot : Module.Finite R (⊥ : Submodule R M) := iff_fg.mpr fg_bot


instance top [Module.Finite R M] : Module.Finite R (⊤ : Submodule R M) := iff_fg.mpr out


/-- The submodule generated by a finite set is `R`-finite. -/
theorem span_of_finite {A : Set M} (hA : Set.Finite A) :
    Module.Finite R (Submodule.span R A) :=
  ⟨(Submodule.fg_top _).mpr ⟨hA.toFinset, hA.coe_toFinset.symm ▸ rfl⟩⟩


/-- The submodule generated by a single element is `R`-finite. -/
instance span_singleton (x : M) : Module.Finite R (R ∙ x) :=
  Module.Finite.span_of_finite R <| Set.finite_singleton _


/-- The submodule generated by a finset is `R`-finite. -/
instance span_finset (s : Finset M) : Module.Finite R (span R (s : Set M)) :=
  ⟨(Submodule.fg_top _).mpr ⟨s, rfl⟩⟩


theorem trans {R : Type*} (A M : Type*) [Semiring R] [Semiring A] [Module R A]
    [AddCommMonoid M] [Module R M] [Module A M] [IsScalarTower R A M] :
    ∀ [Module.Finite R A] [Module.Finite A M], Module.Finite R M
  | ⟨⟨s, hs⟩⟩, ⟨⟨t, ht⟩⟩ =>
    ⟨Submodule.fg_def.2
        ⟨Set.image2 (· • ·) (↑s : Set A) (↑t : Set M),
          Set.Finite.image2 _ s.finite_toSet t.finite_toSet, by
          erw [Set.image2_smul, Submodule.span_smul_of_span_eq_top hs (↑t : Set M), ht,
            Submodule.restrictScalars_top]⟩⟩


lemma of_equiv_equiv {A₁ B₁ A₂ B₂ : Type*} [CommRing A₁] [CommRing B₁]
    [CommRing A₂] [CommRing B₂] [Algebra A₁ B₁] [Algebra A₂ B₂] (e₁ : A₁ ≃+* A₂) (e₂ : B₁ ≃+* B₂)
    (he : RingHom.comp (algebraMap A₂ B₂) ↑e₁ = RingHom.comp ↑e₂ (algebraMap A₁ B₁))
    [Module.Finite A₁ B₁] : Module.Finite A₂ B₂ := by
  /-
    A₁ : Type u_6
    B₁ : Type u_7
    A₂ : Type u_8
    B₂ : Type u_9
    inst✝⁶ : CommRing A₁
    inst✝⁵ : CommRing B₁
    inst✝⁴ : CommRing A₂
    inst✝³ : CommRing B₂
    inst✝² : Algebra A₁ B₁
    inst✝¹ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    inst✝ : Module.Finite A₁ B₁
    ⊢ Module.Finite A₂ B₂
  -/
  letI := e₁.toRingHom.toAlgebra
  /-
    A₁ : Type u_6
    B₁ : Type u_7
    A₂ : Type u_8
    B₂ : Type u_9
    inst✝⁶ : CommRing A₁
    inst✝⁵ : CommRing B₁
    inst✝⁴ : CommRing A₂
    inst✝³ : CommRing B₂
    inst✝² : Algebra A₁ B₁
    inst✝¹ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    inst✝ : Module.Finite A₁ B₁
    this : Algebra A₁ A₂ := e₁.toRingHom.toAlgebra
    ⊢ Module.Finite A₂ B₂
  -/
  letI := ((algebraMap A₁ B₁).comp e₁.symm.toRingHom).toAlgebra
  haveI : IsScalarTower A₁ A₂ B₁ := IsScalarTower.of_algebraMap_eq
    (fun x ↦ by simp [RingHom.algebraMap_toAlgebra])
  let e : B₁ ≃ₐ[A₂] B₂ :=
    { e₂ with
      commutes' := fun r ↦ by
        simpa [RingHom.algebraMap_toAlgebra] using DFunLike.congr_fun he.symm (e₁.symm r) }
  /-
    A₁ : Type u_6
    B₁ : Type u_7
    A₂ : Type u_8
    B₂ : Type u_9
    inst✝⁶ : CommRing A₁
    inst✝⁵ : CommRing B₁
    inst✝⁴ : CommRing A₂
    inst✝³ : CommRing B₂
    inst✝² : Algebra A₁ B₁
    inst✝¹ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    inst✝ : Module.Finite A₁ B₁
    this✝¹ : Algebra A₁ A₂ := e₁.toRingHom.toAlgebra
    this✝ : Algebra A₂ B₁ := ((algebraMap A₁ B₁).comp e₁.symm.toRingHom).toAlgebra
    this : IsScalarTower A₁ A₂ B₁
    e : AlgEquiv A₂ B₁ B₂ := { toEquiv := e₂.toEquiv, map_mul' := ⋯, map_add' := ⋯ …
    ⊢ Module.Finite A₂ B₂
  -/
  haveI := Module.Finite.of_restrictScalars_finite A₁ A₂ B₁
  /-
    A₁ : Type u_6
    B₁ : Type u_7
    A₂ : Type u_8
    B₂ : Type u_9
    inst✝⁶ : CommRing A₁
    inst✝⁵ : CommRing B₁
    inst✝⁴ : CommRing A₂
    inst✝³ : CommRing B₂
    inst✝² : Algebra A₁ B₁
    inst✝¹ : Algebra A₂ B₂
    e₁ : RingEquiv A₁ A₂
    e₂ : RingEquiv B₁ B₂
    he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
    inst✝ : Module.Finite A₁ B₁
    this✝² : Algebra A₁ A₂ := e₁.toRingHom.toAlgebra
    this✝¹ : Algebra A₂ B₁ := ((algebraMap A₁ B₁).comp e₁.symm.toRingHom).toAlgebra
    this✝ : IsScalarTower A₁ A₂ B₁
    e : AlgEquiv A₂ B₁ B₂ := { toEquiv := e₂.toEquiv, map_mul' := ⋯, map_add' := ⋯ …
    this : Module.Finite A₂ B₁
    ⊢ Module.Finite A₂ B₂
  -/
  exact Module.Finite.equiv e.toLinearEquiv
  /-
    🎉 no goals
  -/


/-- The sup of two fg submodules is finite. Also see `Submodule.FG.sup`. -/
instance finite_sup (S₁ S₂ : Submodule R V) [h₁ : Module.Finite R S₁]
    [h₂ : Module.Finite R S₂] : Module.Finite R (S₁ ⊔ S₂ : Submodule R V) := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    S₁ S₂ : Submodule R V
    h₁ : Module.Finite R (Subtype fun x => Membership.mem S₁ x)
    h₂ : Module.Finite R (Subtype fun x => Membership.mem S₂ x)
    ⊢ Module.Finite R (Subtype fun x => Membership.mem (Max.max S₁ S₂) x)
  -/
  rw [finite_def] at *
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    S₁ S₂ : Submodule R V
    h₁ : Top.top.FG
    h₂ : Top.top.FG
    ⊢ Top.top.FG
  -/
  exact (fg_top _).2 (((fg_top S₁).1 h₁).sup ((fg_top S₂).1 h₂))
  /-
    🎉 no goals
  -/


/-- The submodule generated by a finite supremum of finite dimensional submodules is
finite-dimensional.

Note that strictly this only needs `∀ i ∈ s, FiniteDimensional K (S i)`, but that doesn't
work well with typeclass search. -/
instance finite_finset_sup {ι : Type*} (s : Finset ι) (S : ι → Submodule R V)
    [∀ i, Module.Finite R (S i)] : Module.Finite R (s.sup S : Submodule R V) := by
  refine
    @Finset.sup_induction _ _ _ _ s S (fun i => Module.Finite R ↑i) (Module.Finite.bot R V)
      ?_ fun i _ => by infer_instance
  /-
    R : Type u_2
    V : Type u_3
    inst✝³ : Ring R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    ι : Type u_1
    s : Finset ι
    S : ι → Submodule R V
    inst✝ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (S i) x)
    ⊢ ∀ (a₁ : Submodule R V), (fun i => Module.Finite R (Subtype fun x => Membersh …
  -/
  intro S₁ hS₁ S₂ hS₂
  /-
    R : Type u_2
    V : Type u_3
    inst✝³ : Ring R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    ι : Type u_1
    s : Finset ι
    S : ι → Submodule R V
    inst✝ : ∀ (i : ι), Module.Finite R (Subtype fun x => Membership.mem (S i) x)
    S₁ : Submodule R V
    hS₁ : Module.Finite R (Subtype fun x => Membership.mem S₁ x)
    S₂ : Submodule R V
    hS₂ : Module.Finite R (Subtype fun x => Membership.mem S₂ x)
    ⊢ Module.Finite R (Subtype fun x => Membership.mem (Max.max S₁ S₂) x)
  -/
  exact Submodule.finite_sup S₁ S₂
  /-
    🎉 no goals
  -/


theorem id : Finite (RingHom.id A) :=
  Module.Finite.self A


theorem of_surjective (f : A →+* B) (hf : Surjective f) : f.Finite :=
  letI := f.toAlgebra
  Module.Finite.of_surjective (Algebra.linearMap A B) hf


theorem comp {g : B →+* C} {f : A →+* B} (hg : g.Finite) (hf : f.Finite) : (g.comp f).Finite := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.Finite
    hf : f.Finite
    ⊢ (g.comp f).Finite
  -/
  algebraize [f, g, g.comp f]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    g : RingHom B C
    f : RingHom A B
    hg : g.Finite
    hf : f.Finite
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Module.Finite B C
    algebraizeInst✝ : Module.Finite A B
    ⊢ (g.comp f).Finite
  -/
  exact Module.Finite.trans B C
  /-
    🎉 no goals
  -/


theorem of_comp_finite {f : A →+* B} {g : B →+* C} (h : (g.comp f).Finite) : g.Finite := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    h : (g.comp f).Finite
    ⊢ g.Finite
  -/
  algebraize [f, g, g.comp f]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : CommRing C
    f : RingHom A B
    g : RingHom B C
    h : (g.comp f).Finite
    algInst✝² : Algebra A B := f.toAlgebra
    algInst✝¹ : Algebra B C := g.toAlgebra
    algInst✝ : Algebra A C := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower A B C := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝ : Module.Finite A C
    ⊢ g.Finite
  -/
  exact Module.Finite.of_restrictScalars_finite A B C
  /-
    🎉 no goals
  -/


theorem id : Finite (AlgHom.id R A) :=
  RingHom.Finite.id A


theorem comp {g : B →ₐ[R] C} {f : A →ₐ[R] B} (hg : g.Finite) (hf : f.Finite) : (g.comp f).Finite :=
  RingHom.Finite.comp hg hf


theorem of_surjective (f : A →ₐ[R] B) (hf : Surjective f) : f.Finite :=
  RingHom.Finite.of_surjective f.toRingHom hf


theorem of_comp_finite {f : A →ₐ[R] B} {g : B →ₐ[R] C} (h : (g.comp f).Finite) : g.Finite :=
  RingHom.Finite.of_comp_finite h


