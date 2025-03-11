/-- A topological space is called a *regular space* if for any closed set `s` and `a ∉ s`, there
exist disjoint open sets `U ⊇ s` and `V ∋ a`. We formulate this condition in terms of `Disjoint`ness
of filters `𝓝ˢ s` and `𝓝 a`. -/
@[mk_iff]
class RegularSpace (X : Type u) [TopologicalSpace X] : Prop where
  /-- If `a` is a point that does not belong to a closed set `s`, then `a` and `s` admit disjoint
  neighborhoods. -/
  regular : ∀ {s : Set X} {a}, IsClosed s → a ∉ s → Disjoint (𝓝ˢ s) (𝓝 a)


theorem regularSpace_TFAE (X : Type u) [TopologicalSpace X] :
    List.TFAE [RegularSpace X,
      ∀ (s : Set X) x, x ∉ closure s → Disjoint (𝓝ˢ s) (𝓝 x),
      ∀ (x : X) (s : Set X), Disjoint (𝓝ˢ s) (𝓝 x) ↔ x ∉ closure s,
      ∀ (x : X) (s : Set X), s ∈ 𝓝 x → ∃ t ∈ 𝓝 x, IsClosed t ∧ t ⊆ s,
      ∀ x : X, (𝓝 x).lift' closure ≤ 𝓝 x,
      ∀ x : X , (𝓝 x).lift' closure = 𝓝 x] := by
  tfae_have 1 ↔ 5 := by
    rw [regularSpace_iff, (@compl_surjective (Set X) _).forall, forall_swap]
    simp only [isClosed_compl_iff, mem_compl_iff, Classical.not_not, @and_comm (_ ∈ _),
      (nhds_basis_opens _).lift'_closure.le_basis_iff (nhds_basis_opens _), and_imp,
      (nhds_basis_opens _).disjoint_iff_right, exists_prop, ← subset_interior_iff_mem_nhdsSet,
      interior_compl, compl_subset_compl]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    tfae_1_iff_5 : Iff (RegularSpace X) (∀ (x : X), LE.le ((nhds x).lift' closure) …
    ⊢ (List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Membersh …
  -/
  tfae_have 5 → 6 := fun h a => (h a).antisymm (𝓝 _).le_lift'_closure
  tfae_have 6 → 4
  | H, a, s, hs => by
    rw [← H] at hs
    rcases (𝓝 a).basis_sets.lift'_closure.mem_iff.mp hs with ⟨U, hU, hUs⟩
    exact ⟨closure U, mem_of_superset hU subset_closure, isClosed_closure, hUs⟩
  tfae_have 4 → 2
  | H, s, a, ha => by
    have ha' : sᶜ ∈ 𝓝 a := by rwa [← mem_interior_iff_mem_nhds, interior_compl]
    rcases H _ _ ha' with ⟨U, hU, hUc, hUs⟩
    refine disjoint_of_disjoint_of_mem disjoint_compl_left ?_ hU
    rwa [← subset_interior_iff_mem_nhdsSet, hUc.isOpen_compl.interior_eq, subset_compl_comm]
  tfae_have 2 → 3 := by
    refine fun H a s => ⟨fun hd has => mem_closure_iff_nhds_ne_bot.mp has ?_, H s a⟩
    exact (hd.symm.mono_right <| @principal_le_nhdsSet _ _ s).eq_bot
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    tfae_1_iff_5 : Iff (RegularSpace X) (∀ (x : X), LE.le ((nhds x).lift' closure) …
    tfae_5_to_6 : (∀ (x : X), LE.le ((nhds x).lift' closure) (nhds x)) → ∀ (x : X) …
    tfae_6_to_4 : (∀ (x : X), Eq ((nhds x).lift' closure) (nhds x)) → ∀ (x : X) (s …
    tfae_4_to_2 : (∀ (x : X) (s : Set X), Membership.mem (nhds x) s → Exists fun t …
    tfae_2_to_3 : (∀ (s : Set X) (x : X), Not (Membership.mem (closure s) x) → Dis …
    ⊢ (List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Membersh …
  -/
  tfae_have 3 → 1 := fun H => ⟨fun hs ha => (H _ _).mpr <| hs.closure_eq.symm ▸ ha⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    tfae_1_iff_5 : Iff (RegularSpace X) (∀ (x : X), LE.le ((nhds x).lift' closure) …
    tfae_5_to_6 : (∀ (x : X), LE.le ((nhds x).lift' closure) (nhds x)) → ∀ (x : X) …
    tfae_6_to_4 : (∀ (x : X), Eq ((nhds x).lift' closure) (nhds x)) → ∀ (x : X) (s …
    tfae_4_to_2 : (∀ (x : X) (s : Set X), Membership.mem (nhds x) s → Exists fun t …
    tfae_2_to_3 : (∀ (s : Set X) (x : X), Not (Membership.mem (closure s) x) → Dis …
    tfae_3_to_1 : (∀ (x : X) (s : Set X), Iff (Disjoint (nhdsSet s) (nhds x)) (Not …
    ⊢ (List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Membersh …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem RegularSpace.of_lift'_closure_le (h : ∀ x : X, (𝓝 x).lift' closure ≤ 𝓝 x) :
    RegularSpace X :=
           /-
             X : Type u_1
             inst✝ : TopologicalSpace X
             h : ∀ (x : X), LE.le ((nhds x).lift' closure) (nhds x)
             ⊢ Eq ((List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Memb …
           -/
           /-
             🎉 no goals
           -/
  Iff.mpr ((regularSpace_TFAE X).out 0 4) h
           /-
             🎉 no goals
           -/


theorem RegularSpace.of_lift'_closure (h : ∀ x : X, (𝓝 x).lift' closure = 𝓝 x) : RegularSpace X :=
           /-
             X : Type u_1
             inst✝ : TopologicalSpace X
             h : ∀ (x : X), Eq ((nhds x).lift' closure) (nhds x)
             ⊢ Eq ((List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Memb …
           -/
           /-
             🎉 no goals
           -/
  Iff.mpr ((regularSpace_TFAE X).out 0 5) h
           /-
             🎉 no goals
           -/


@[deprecated (since := "2024-02-28")]
alias RegularSpace.ofLift'_closure := RegularSpace.of_lift'_closure


theorem RegularSpace.of_hasBasis {ι : X → Sort*} {p : ∀ a, ι a → Prop} {s : ∀ a, ι a → Set X}
    (h₁ : ∀ a, (𝓝 a).HasBasis (p a) (s a)) (h₂ : ∀ a i, p a i → IsClosed (s a i)) :
    RegularSpace X :=
  .of_lift'_closure fun a => (h₁ a).lift'_closure_eq_self (h₂ a)


@[deprecated (since := "2024-02-28")]
alias RegularSpace.ofBasis := RegularSpace.of_hasBasis


theorem RegularSpace.of_exists_mem_nhds_isClosed_subset
    (h : ∀ (x : X), ∀ s ∈ 𝓝 x, ∃ t ∈ 𝓝 x, IsClosed t ∧ t ⊆ s) : RegularSpace X :=
           /-
             X : Type u_1
             inst✝ : TopologicalSpace X
             h : ∀ (x : X) (s : Set X), Membership.mem (nhds x) s → Exists fun t => And (Me …
             ⊢ Eq ((List.cons (RegularSpace X) (List.cons (∀ (s : Set X) (x : X), Not (Memb …
           -/
           /-
             🎉 no goals
           -/
  Iff.mpr ((regularSpace_TFAE X).out 0 3) h
           /-
             🎉 no goals
           -/


@[deprecated (since := "2024-02-28")]
alias RegularSpace.ofExistsMemNhdsIsClosedSubset := RegularSpace.of_exists_mem_nhds_isClosed_subset


/-- A weakly locally compact R₁ space is regular. -/
instance (priority := 100) [WeaklyLocallyCompactSpace X] [R1Space X] : RegularSpace X :=
  .of_hasBasis isCompact_isClosed_basis_nhds fun _ _ ⟨_, _, h⟩ ↦ h


theorem disjoint_nhdsSet_nhds : Disjoint (𝓝ˢ s) (𝓝 x) ↔ x ∉ closure s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    x : X
    s : Set X
    ⊢ Iff (Disjoint (nhdsSet s) (nhds x)) (Not (Membership.mem (closure s) x))
  -/
  have h := (regularSpace_TFAE X).out 0 2
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    x : X
    s : Set X
    h : Iff (RegularSpace X) (∀ (x : X) (s : Set X), Iff (Disjoint (nhdsSet s) (nh …
    ⊢ Iff (Disjoint (nhdsSet s) (nhds x)) (Not (Membership.mem (closure s) x))
  -/
  exact h.mp ‹_› _ _
  /-
    🎉 no goals
  -/


theorem disjoint_nhds_nhdsSet : Disjoint (𝓝 x) (𝓝ˢ s) ↔ x ∉ closure s :=
  disjoint_comm.trans disjoint_nhdsSet_nhds


/-- A regular space is R₁. -/
instance (priority := 100) : R1Space X where
  specializes_or_disjoint_nhds _ _ := or_iff_not_imp_left.2 fun h ↦ by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : RegularSpace X
      x : X
      s : Set X
      x✝¹ x✝ : X
      h : Not (Specializes x✝¹ x✝)
      ⊢ Disjoint (nhds x✝¹) (nhds x✝)
    -/
    rwa [← nhdsSet_singleton, disjoint_nhdsSet_nhds, ← specializes_iff_mem_closure]
    /-
      🎉 no goals
    -/


theorem exists_mem_nhds_isClosed_subset {x : X} {s : Set X} (h : s ∈ 𝓝 x) :
    ∃ t ∈ 𝓝 x, IsClosed t ∧ t ⊆ s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    x : X
    s : Set X
    h : Membership.mem (nhds x) s
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (And (IsClosed t) (HasSubset …
  -/
  have h' := (regularSpace_TFAE X).out 0 3
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    x : X
    s : Set X
    h : Membership.mem (nhds x) s
    h' : Iff (RegularSpace X) (∀ (x : X) (s : Set X), Membership.mem (nhds x) s →  …
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (And (IsClosed t) (HasSubset …
  -/
  exact h'.mp ‹_› _ _ h
  /-
    🎉 no goals
  -/


theorem closed_nhds_basis (x : X) : (𝓝 x).HasBasis (fun s : Set X => s ∈ 𝓝 x ∧ IsClosed s) id :=
  hasBasis_self.2 fun _ => exists_mem_nhds_isClosed_subset


theorem lift'_nhds_closure (x : X) : (𝓝 x).lift' closure = 𝓝 x :=
  (closed_nhds_basis x).lift'_closure_eq_self fun _ => And.right


theorem Filter.HasBasis.nhds_closure {ι : Sort*} {x : X} {p : ι → Prop} {s : ι → Set X}
    (h : (𝓝 x).HasBasis p s) : (𝓝 x).HasBasis p fun i => closure (s i) :=
  lift'_nhds_closure x ▸ h.lift'_closure


theorem hasBasis_nhds_closure (x : X) : (𝓝 x).HasBasis (fun s => s ∈ 𝓝 x) closure :=
  (𝓝 x).basis_sets.nhds_closure


theorem hasBasis_opens_closure (x : X) : (𝓝 x).HasBasis (fun s => x ∈ s ∧ IsOpen s) closure :=
  (nhds_basis_opens x).nhds_closure


theorem IsCompact.exists_isOpen_closure_subset {K U : Set X} (hK : IsCompact K) (hU : U ∈ 𝓝ˢ K) :
    ∃ V, IsOpen V ∧ K ⊆ V ∧ closure V ⊆ U := by
  have hd : Disjoint (𝓝ˢ K) (𝓝ˢ Uᶜ) := by
    simpa [hK.disjoint_nhdsSet_left, disjoint_nhds_nhdsSet,
      ← subset_interior_iff_mem_nhdsSet] using hU
  rcases ((hasBasis_nhdsSet _).disjoint_iff (hasBasis_nhdsSet _)).1 hd
    with ⟨V, ⟨hVo, hKV⟩, W, ⟨hW, hUW⟩, hVW⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    K U : Set X
    hK : IsCompact K
    hU : Membership.mem (nhdsSet K) U
    hd : Disjoint (nhdsSet K) (nhdsSet (HasCompl.compl U))
    V : Set X
    hVo : IsOpen V
    hKV : HasSubset.Subset K V
    W : Set X
    hVW : Disjoint V W
    hW : IsOpen W
    hUW : HasSubset.Subset (HasCompl.compl U) W
    ⊢ Exists fun V => And (IsOpen V) (And (HasSubset.Subset K V) (HasSubset.Subset …
  -/
  refine ⟨V, hVo, hKV, Subset.trans ?_ (compl_subset_comm.1 hUW)⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    K U : Set X
    hK : IsCompact K
    hU : Membership.mem (nhdsSet K) U
    hd : Disjoint (nhdsSet K) (nhdsSet (HasCompl.compl U))
    V : Set X
    hVo : IsOpen V
    hKV : HasSubset.Subset K V
    W : Set X
    hVW : Disjoint V W
    hW : IsOpen W
    hUW : HasSubset.Subset (HasCompl.compl U) W
    ⊢ HasSubset.Subset (closure V) (HasCompl.compl W)
  -/
  exact closure_minimal hVW.subset_compl_right hW.isClosed_compl
  /-
    🎉 no goals
  -/


theorem IsCompact.lift'_closure_nhdsSet {K : Set X} (hK : IsCompact K) :
    (𝓝ˢ K).lift' closure = 𝓝ˢ K := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    K : Set X
    hK : IsCompact K
    ⊢ Eq ((nhdsSet K).lift' closure) (nhdsSet K)
  -/
  refine le_antisymm (fun U hU ↦ ?_) (le_lift'_closure _)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    K : Set X
    hK : IsCompact K
    U : Set X
    hU : Membership.mem (nhdsSet K) U
    ⊢ Membership.mem ((nhdsSet K).lift' closure) U
  -/
  rcases hK.exists_isOpen_closure_subset hU with ⟨V, hVo, hKV, hVU⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    K : Set X
    hK : IsCompact K
    U : Set X
    hU : Membership.mem (nhdsSet K) U
    V : Set X
    hVo : IsOpen V
    hKV : HasSubset.Subset K V
    hVU : HasSubset.Subset (closure V) U
    ⊢ Membership.mem ((nhdsSet K).lift' closure) U
  -/
  exact mem_of_superset (mem_lift' <| hVo.mem_nhdsSet.2 hKV) hVU
  /-
    🎉 no goals
  -/


theorem TopologicalSpace.IsTopologicalBasis.nhds_basis_closure {B : Set (Set X)}
    (hB : IsTopologicalBasis B) (x : X) :
    (𝓝 x).HasBasis (fun s : Set X => x ∈ s ∧ s ∈ B) closure := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    B : Set (Set X)
    hB : TopologicalSpace.IsTopologicalBasis B
    x : X
    ⊢ (nhds x).HasBasis (fun s => And (Membership.mem s x) (Membership.mem B s)) c …
  -/
  simpa only [and_comm] using hB.nhds_hasBasis.nhds_closure
  /-
    🎉 no goals
  -/


theorem TopologicalSpace.IsTopologicalBasis.exists_closure_subset {B : Set (Set X)}
    (hB : IsTopologicalBasis B) {x : X} {s : Set X} (h : s ∈ 𝓝 x) :
    ∃ t ∈ B, x ∈ t ∧ closure t ⊆ s := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    B : Set (Set X)
    hB : TopologicalSpace.IsTopologicalBasis B
    x : X
    s : Set X
    h : Membership.mem (nhds x) s
    ⊢ Exists fun t => And (Membership.mem B t) (And (Membership.mem t x) (HasSubse …
  -/
  simpa only [exists_prop, and_assoc] using hB.nhds_hasBasis.nhds_closure.mem_iff.mp h
  /-
    🎉 no goals
  -/


protected theorem Topology.IsInducing.regularSpace [TopologicalSpace Y] {f : Y → X}
    (hf : IsInducing f) : RegularSpace Y :=
  .of_hasBasis
                 /-
                   X : Type u_1
                   Y : Type u_2
                   inst✝² : TopologicalSpace X
                   inst✝¹ : RegularSpace X
                   inst✝ : TopologicalSpace Y
                   f : Y → X
                   hf : Topology.IsInducing f
                   b : Y
                   ⊢ (nhds b).HasBasis (?m.14485 b) (?m.14486 b)
                 -/
    (fun b => by rw [hf.nhds_eq_comap b]; exact (closed_nhds_basis _).comap _)
                                          /-
                                            🎉 no goals
                                          -/
                     /-
                       X : Type u_1
                       Y : Type u_2
                       inst✝² : TopologicalSpace X
                       inst✝¹ : RegularSpace X
                       inst✝ : TopologicalSpace Y
                       f : Y → X
                       hf : Topology.IsInducing f
                       b : Y
                       s : Set X
                       hs : And (Membership.mem (nhds (f b)) s) (IsClosed s)
                       ⊢ IsClosed (Set.preimage f (id s))
                     -/
    fun b s hs => by exact hs.2.preimage hf.continuous
                     /-
                       🎉 no goals
                     -/


@[deprecated (since := "2024-10-28")] alias Inducing.regularSpace := IsInducing.regularSpace


theorem regularSpace_induced (f : Y → X) : @RegularSpace Y (induced f ‹_›) :=
  letI := induced f ‹_›
  (IsInducing.induced f).regularSpace


theorem regularSpace_sInf {X} {T : Set (TopologicalSpace X)} (h : ∀ t ∈ T, @RegularSpace X t) :
    @RegularSpace X (sInf T) := by
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    h : ∀ (t : TopologicalSpace X), Membership.mem T t → RegularSpace X
    ⊢ RegularSpace X
  -/
  let _ := sInf T
  have : ∀ a, (𝓝 a).HasBasis
      (fun If : Σ I : Set T, I → Set X =>
        If.1.Finite ∧ ∀ i : If.1, If.2 i ∈ @nhds X i a ∧ @IsClosed X i (If.2 i))
      fun If => ⋂ i : If.1, If.snd i := fun a ↦ by
    rw [nhds_sInf, ← iInf_subtype'']
    exact hasBasis_iInf fun t : T => @closed_nhds_basis X t (h t t.2) a
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    h : ∀ (t : TopologicalSpace X), Membership.mem T t → RegularSpace X
    x✝ : TopologicalSpace X := InfSet.sInf T
    this : ∀ (a : X), (nhds a).HasBasis (fun If => And If.fst.Finite (∀ (i : ↑If.f …
    ⊢ RegularSpace X
  -/
  refine .of_hasBasis this fun a If hIf => isClosed_iInter fun i => ?_
  /-
    X : Type u_3
    T : Set (TopologicalSpace X)
    h : ∀ (t : TopologicalSpace X), Membership.mem T t → RegularSpace X
    x✝ : TopologicalSpace X := InfSet.sInf T
    this : ∀ (a : X), (nhds a).HasBasis (fun If => And If.fst.Finite (∀ (i : ↑If.f …
    a : X
    If : Sigma fun I => ↑I → Set X
    hIf : And If.fst.Finite (∀ (i : ↑If.fst), And (Membership.mem (nhds a) (If.snd …
    i : ↑If.fst
    ⊢ IsClosed (If.snd i)
  -/
  exact (hIf.2 i).2.mono (sInf_le (i : T).2)
  /-
    🎉 no goals
  -/


theorem regularSpace_iInf {ι X} {t : ι → TopologicalSpace X} (h : ∀ i, @RegularSpace X (t i)) :
    @RegularSpace X (iInf t) :=
  regularSpace_sInf <| forall_mem_range.mpr h


theorem RegularSpace.inf {X} {t₁ t₂ : TopologicalSpace X} (h₁ : @RegularSpace X t₁)
    (h₂ : @RegularSpace X t₂) : @RegularSpace X (t₁ ⊓ t₂) := by
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : RegularSpace X
    h₂ : RegularSpace X
    ⊢ RegularSpace X
  -/
  rw [inf_eq_iInf]
  /-
    X : Type u_3
    t₁ t₂ : TopologicalSpace X
    h₁ : RegularSpace X
    h₂ : RegularSpace X
    ⊢ RegularSpace X
  -/
  exact regularSpace_iInf (Bool.forall_bool.2 ⟨h₂, h₁⟩)
  /-
    🎉 no goals
  -/


instance {p : X → Prop} : RegularSpace (Subtype p) :=
  IsEmbedding.subtypeVal.isInducing.regularSpace


instance [TopologicalSpace Y] [RegularSpace Y] : RegularSpace (X × Y) :=
  (regularSpace_induced (@Prod.fst X Y)).inf (regularSpace_induced (@Prod.snd X Y))


instance {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)] [∀ i, RegularSpace (X i)] :
    RegularSpace (∀ i, X i) :=
  regularSpace_iInf fun _ => regularSpace_induced _


/-- In a regular space, if a compact set and a closed set are disjoint, then they have disjoint
neighborhoods. -/
lemma SeparatedNhds.of_isCompact_isClosed {s t : Set X}
    (hs : IsCompact s) (ht : IsClosed t) (hst : Disjoint s t) : SeparatedNhds s t := by
  simpa only [separatedNhds_iff_disjoint, hs.disjoint_nhdsSet_left, disjoint_nhds_nhdsSet,
    ht.closure_eq, disjoint_left] using hst


/-- This technique to witness `HasSeparatingCover` in regular Lindelöf topological spaces
will be used to prove regular Lindelöf spaces are normal. -/
lemma IsClosed.HasSeparatingCover {s t : Set X} [LindelofSpace X] [RegularSpace X]
    (s_cl : IsClosed s) (t_cl : IsClosed t) (st_dis : Disjoint s t) : HasSeparatingCover s t := by
  -- `IsLindelof.indexed_countable_subcover` requires the space be Nonempty
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    s t : Set X
    inst✝¹ : LindelofSpace X
    inst✝ : RegularSpace X
    s_cl : IsClosed s
    t_cl : IsClosed t
    st_dis : Disjoint s t
    ⊢ _root_.HasSeparatingCover s t
  -/
  rcases isEmpty_or_nonempty X with empty_X | nonempty_X
    /-
      case inl
      X : Type u_1
      inst✝² : TopologicalSpace X
      s t : Set X
      inst✝¹ : LindelofSpace X
      inst✝ : RegularSpace X
      s_cl : IsClosed s
      t_cl : IsClosed t
      st_dis : Disjoint s t
      empty_X : IsEmpty X
      ⊢ _root_.HasSeparatingCover s t
    -/
  · rw [subset_eq_empty (t := s) (fun ⦃_⦄ _ ↦ trivial) (univ_eq_empty_iff.mpr empty_X)]
    /-
      case inl
      X : Type u_1
      inst✝² : TopologicalSpace X
      s t : Set X
      inst✝¹ : LindelofSpace X
      inst✝ : RegularSpace X
      s_cl : IsClosed s
      t_cl : IsClosed t
      st_dis : Disjoint s t
      empty_X : IsEmpty X
      ⊢ _root_.HasSeparatingCover EmptyCollection.emptyCollection t
    -/
    exact hasSeparatingCovers_iff_separatedNhds.mpr (SeparatedNhds.empty_left t) |>.1
    /-
      🎉 no goals
    -/
  -- This is almost `HasSeparatingCover`, but is not countable. We define for all `a : X` for use
  -- with `IsLindelof.indexed_countable_subcover` momentarily.
  have (a : X) : ∃ n : Set X, IsOpen n ∧ Disjoint (closure n) t ∧ (a ∈ s → a ∈ n) := by
    wlog ains : a ∈ s
    · exact ⟨∅, isOpen_empty, SeparatedNhds.empty_left t |>.disjoint_closure_left, fun a ↦ ains a⟩
    obtain ⟨n, nna, ncl, nsubkc⟩ := ((regularSpace_TFAE X).out 0 3 :).mp ‹RegularSpace X› a tᶜ <|
      t_cl.compl_mem_nhds (disjoint_left.mp st_dis ains)
    exact
      ⟨interior n,
       isOpen_interior,
       disjoint_left.mpr fun ⦃_⦄ ain ↦
         nsubkc <| (IsClosed.closure_subset_iff ncl).mpr interior_subset ain,
       fun _ ↦ mem_interior_iff_mem_nhds.mpr nna⟩
  -- By Lindelöf, we may obtain a countable subcover witnessing `HasSeparatingCover`
  /-
    case inr
    X : Type u_1
    inst✝² : TopologicalSpace X
    s t : Set X
    inst✝¹ : LindelofSpace X
    inst✝ : RegularSpace X
    s_cl : IsClosed s
    t_cl : IsClosed t
    st_dis : Disjoint s t
    nonempty_X : Nonempty X
    this : ∀ (a : X), Exists fun n => And (IsOpen n) (And (Disjoint (closure n) t) …
    ⊢ _root_.HasSeparatingCover s t
  -/
  choose u u_open u_dis u_nhd using this
  obtain ⟨f, f_cov⟩ := s_cl.isLindelof.indexed_countable_subcover
    u u_open (fun a ainh ↦ mem_iUnion.mpr ⟨a, u_nhd a ainh⟩)
  /-
    case inr.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    s t : Set X
    inst✝¹ : LindelofSpace X
    inst✝ : RegularSpace X
    s_cl : IsClosed s
    t_cl : IsClosed t
    st_dis : Disjoint s t
    nonempty_X : Nonempty X
    u : X → Set X
    u_open : ∀ (a : X), IsOpen (u a)
    u_dis : ∀ (a : X), Disjoint (closure (u a)) t
    u_nhd : ∀ (a : X), Membership.mem s a → Membership.mem (u a) a
    f : Nat → X
    f_cov : HasSubset.Subset s (Set.iUnion fun n => u (f n))
    ⊢ _root_.HasSeparatingCover s t
  -/
  exact ⟨u ∘ f, f_cov, fun n ↦ ⟨u_open (f n), u_dis (f n)⟩⟩
  /-
    🎉 no goals
  -/



/-- In a (possibly non-Hausdorff) locally compact regular space, for every containment `K ⊆ U` of
  a compact set `K` in an open set `U`, there is a compact closed neighborhood `L`
  such that `K ⊆ L ⊆ U`: equivalently, there is a compact closed set `L` such
  that `K ⊆ interior L` and `L ⊆ U`. -/
theorem exists_compact_closed_between [LocallyCompactSpace X] [RegularSpace X]
    {K U : Set X} (hK : IsCompact K) (hU : IsOpen U) (h_KU : K ⊆ U) :
    ∃ L, IsCompact L ∧ IsClosed L ∧ K ⊆ interior L ∧ L ⊆ U :=
  let ⟨L, L_comp, KL, LU⟩ := exists_compact_between hK hU h_KU
  ⟨closure L, L_comp.closure, isClosed_closure, KL.trans <| interior_mono subset_closure,
    L_comp.closure_subset_of_isOpen hU LU⟩


/-- In a locally compact regular space, given a compact set `K` inside an open set `U`, we can find
an open set `V` between these sets with compact closure: `K ⊆ V` and the closure of `V` is
inside `U`. -/
theorem exists_open_between_and_isCompact_closure [LocallyCompactSpace X] [RegularSpace X]
    {K U : Set X} (hK : IsCompact K) (hU : IsOpen U) (hKU : K ⊆ U) :
    ∃ V, IsOpen V ∧ K ⊆ V ∧ closure V ⊆ U ∧ IsCompact (closure V) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : RegularSpace X
    K U : Set X
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    ⊢ Exists fun V => And (IsOpen V) (And (HasSubset.Subset K V) (And (HasSubset.S …
  -/
  rcases exists_compact_closed_between hK hU hKU with ⟨L, L_compact, L_closed, KL, LU⟩
  have A : closure (interior L) ⊆ L := by
    apply (closure_mono interior_subset).trans (le_of_eq L_closed.closure_eq)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : RegularSpace X
    K U : Set X
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    L : Set X
    L_compact : IsCompact L
    L_closed : IsClosed L
    KL : HasSubset.Subset K (interior L)
    LU : HasSubset.Subset L U
    A : HasSubset.Subset (closure (interior L)) L
    ⊢ Exists fun V => And (IsOpen V) (And (HasSubset.Subset K V) (And (HasSubset.S …
  -/
  refine ⟨interior L, isOpen_interior, KL, A.trans LU, ?_⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : RegularSpace X
    K U : Set X
    hK : IsCompact K
    hU : IsOpen U
    hKU : HasSubset.Subset K U
    L : Set X
    L_compact : IsCompact L
    L_closed : IsClosed L
    KL : HasSubset.Subset K (interior L)
    LU : HasSubset.Subset L U
    A : HasSubset.Subset (closure (interior L)) L
    ⊢ IsCompact (closure (interior L))
  -/
  exact L_compact.closure_of_subset interior_subset
  /-
    🎉 no goals
  -/


/-- A T₂.₅ space, also known as a Urysohn space, is a topological space
  where for every pair `x ≠ y`, there are two open sets, with the intersection of closures
  empty, one containing `x` and the other `y` . -/
class T25Space (X : Type u) [TopologicalSpace X] : Prop where
  /-- Given two distinct points in a T₂.₅ space, their filters of closed neighborhoods are
  disjoint. -/
  t2_5 : ∀ ⦃x y : X⦄, x ≠ y → Disjoint ((𝓝 x).lift' closure) ((𝓝 y).lift' closure)


@[simp]
theorem disjoint_lift'_closure_nhds [T25Space X] {x y : X} :
    Disjoint ((𝓝 x).lift' closure) ((𝓝 y).lift' closure) ↔ x ≠ y :=
                   /-
                     X : Type u_1
                     inst✝¹ : TopologicalSpace X
                     inst✝ : T25Space X
                     x y : X
                     h : Disjoint ((nhds x).lift' closure) ((nhds y).lift' closure)
                     hxy : Eq x y
                     ⊢ False
                   -/
  ⟨fun h hxy => by simp [hxy, nhds_neBot.ne] at h, fun h => T25Space.t2_5 h⟩
                   /-
                     🎉 no goals
                   -/

-- see Note [lower instance priority]

instance (priority := 100) T25Space.t2Space [T25Space X] : T2Space X :=
  t2Space_iff_disjoint_nhds.2 fun _ _ hne =>
    (disjoint_lift'_closure_nhds.2 hne).mono (le_lift'_closure _) (le_lift'_closure _)


theorem exists_nhds_disjoint_closure [T25Space X] {x y : X} (h : x ≠ y) :
    ∃ s ∈ 𝓝 x, ∃ t ∈ 𝓝 y, Disjoint (closure s) (closure t) :=
  ((𝓝 x).basis_sets.lift'_closure.disjoint_iff (𝓝 y).basis_sets.lift'_closure).1 <|
    disjoint_lift'_closure_nhds.2 h


theorem exists_open_nhds_disjoint_closure [T25Space X] {x y : X} (h : x ≠ y) :
    ∃ u : Set X,
      x ∈ u ∧ IsOpen u ∧ ∃ v : Set X, y ∈ v ∧ IsOpen v ∧ Disjoint (closure u) (closure v) := by
  simpa only [exists_prop, and_assoc] using
    ((nhds_basis_opens x).lift'_closure.disjoint_iff (nhds_basis_opens y).lift'_closure).1
      (disjoint_lift'_closure_nhds.2 h)


theorem T25Space.of_injective_continuous [TopologicalSpace Y] [T25Space Y] {f : X → Y}
    (hinj : Injective f) (hcont : Continuous f) : T25Space X where
  t2_5 x y hne := (tendsto_lift'_closure_nhds hcont x).disjoint (t2_5 <| hinj.ne hne)
    (tendsto_lift'_closure_nhds hcont y)


theorem Topology.IsEmbedding.t25Space [TopologicalSpace Y] [T25Space Y] {f : X → Y}
    (hf : IsEmbedding f) : T25Space X :=
  .of_injective_continuous hf.injective hf.continuous


@[deprecated (since := "2024-10-26")]
alias Embedding.t25Space := IsEmbedding.t25Space


instance Subtype.instT25Space [T25Space X] {p : X → Prop} : T25Space {x // p x} :=
  IsEmbedding.subtypeVal.t25Space


/-- A T₃ space is a T₀ space which is a regular space. Any T₃ space is a T₁ space, a T₂ space, and
a T₂.₅ space. -/
class T3Space (X : Type u) [TopologicalSpace X] extends T0Space X, RegularSpace X : Prop


instance (priority := 90) instT3Space [T0Space X] [RegularSpace X] : T3Space X := ⟨⟩


theorem RegularSpace.t3Space_iff_t0Space [RegularSpace X] : T3Space X ↔ T0Space X := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : RegularSpace X
    ⊢ Iff (T3Space X) (T0Space X)
  -/
                            /-
                              🎉 no goals
                            -/
  constructor <;> intro <;> infer_instance
                            /-
                              🎉 no goals
                            -/

-- see Note [lower instance priority]

instance (priority := 100) T3Space.t25Space [T3Space X] : T25Space X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    ⊢ T25Space X
  -/
  refine ⟨fun x y hne => ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    hne : Ne x y
    ⊢ Disjoint ((nhds x).lift' closure) ((nhds y).lift' closure)
  -/
  rw [lift'_nhds_closure, lift'_nhds_closure]
  have : x ∉ closure {y} ∨ y ∉ closure {x} :=
    (t0Space_iff_or_not_mem_closure X).mp inferInstance hne
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    hne : Ne x y
    this : Or (Not (Membership.mem (closure (Singleton.singleton y)) x)) (Not (Mem …
    ⊢ Disjoint (nhds x) (nhds y)
  -/
  simp only [← disjoint_nhds_nhdsSet, nhdsSet_singleton] at this
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    hne : Ne x y
    this : Or (Disjoint (nhds x) (nhds y)) (Disjoint (nhds y) (nhds x))
    ⊢ Disjoint (nhds x) (nhds y)
  -/
  exact this.elim id fun h => h.symm
  /-
    🎉 no goals
  -/


protected theorem Topology.IsEmbedding.t3Space [TopologicalSpace Y] [T3Space Y] {f : X → Y}
    (hf : IsEmbedding f) : T3Space X :=
  { toT0Space := hf.t0Space
    toRegularSpace := hf.isInducing.regularSpace }


@[deprecated (since := "2024-10-26")]
alias Embedding.t3Space := IsEmbedding.t3Space


instance Subtype.t3Space [T3Space X] {p : X → Prop} : T3Space (Subtype p) :=
  IsEmbedding.subtypeVal.t3Space


instance ULift.instT3Space [T3Space X] : T3Space (ULift X) :=
  IsEmbedding.uliftDown.t3Space


instance [TopologicalSpace Y] [T3Space X] [T3Space Y] : T3Space (X × Y) := ⟨⟩


instance {ι : Type*} {X : ι → Type*} [∀ i, TopologicalSpace (X i)] [∀ i, T3Space (X i)] :
    T3Space (∀ i, X i) := ⟨⟩


/-- Given two points `x ≠ y`, we can find neighbourhoods `x ∈ V₁ ⊆ U₁` and `y ∈ V₂ ⊆ U₂`,
with the `Vₖ` closed and the `Uₖ` open, such that the `Uₖ` are disjoint. -/
theorem disjoint_nested_nhds [T3Space X] {x y : X} (h : x ≠ y) :
    ∃ U₁ ∈ 𝓝 x, ∃ V₁ ∈ 𝓝 x, ∃ U₂ ∈ 𝓝 y, ∃ V₂ ∈ 𝓝 y,
      IsClosed V₁ ∧ IsClosed V₂ ∧ IsOpen U₁ ∧ IsOpen U₂ ∧ V₁ ⊆ U₁ ∧ V₂ ⊆ U₂ ∧ Disjoint U₁ U₂ := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    h : Ne x y
    ⊢ Exists fun U₁ => And (Membership.mem (nhds x) U₁) (Exists fun V₁ => And (Mem …
  -/
  rcases t2_separation h with ⟨U₁, U₂, U₁_op, U₂_op, x_in, y_in, H⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    h : Ne x y
    U₁ U₂ : Set X
    U₁_op : IsOpen U₁
    U₂_op : IsOpen U₂
    x_in : Membership.mem U₁ x
    y_in : Membership.mem U₂ y
    H : Disjoint U₁ U₂
    ⊢ Exists fun U₁ => And (Membership.mem (nhds x) U₁) (Exists fun V₁ => And (Mem …
  -/
  rcases exists_mem_nhds_isClosed_subset (U₁_op.mem_nhds x_in) with ⟨V₁, V₁_in, V₁_closed, h₁⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T3Space X
    x y : X
    h : Ne x y
    U₁ U₂ : Set X
    U₁_op : IsOpen U₁
    U₂_op : IsOpen U₂
    x_in : Membership.mem U₁ x
    y_in : Membership.mem U₂ y
    H : Disjoint U₁ U₂
    V₁ : Set X
    V₁_in : Membership.mem (nhds x) V₁
    V₁_closed : IsClosed V₁
    h₁ : HasSubset.Subset V₁ U₁
    ⊢ Exists fun U₁ => And (Membership.mem (nhds x) U₁) (Exists fun V₁ => And (Mem …
  -/
  rcases exists_mem_nhds_isClosed_subset (U₂_op.mem_nhds y_in) with ⟨V₂, V₂_in, V₂_closed, h₂⟩
  exact ⟨U₁, mem_of_superset V₁_in h₁, V₁, V₁_in, U₂, mem_of_superset V₂_in h₂, V₂, V₂_in,
    V₁_closed, V₂_closed, U₁_op, U₂_op, h₁, h₂, H⟩


/-- The `SeparationQuotient` of a regular space is a T₃ space. -/
instance [RegularSpace X] : T3Space (SeparationQuotient X) where
  regular {s a} hs ha := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : RegularSpace X
      s : Set (SeparationQuotient X)
      a : SeparationQuotient X
      hs : IsClosed s
      ha : Not (Membership.mem s a)
      ⊢ Disjoint (nhdsSet s) (nhds a)
    -/
    rcases surjective_mk a with ⟨a, rfl⟩
    /-
      case intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : RegularSpace X
      s : Set (SeparationQuotient X)
      hs : IsClosed s
      a : X
      ha : Not (Membership.mem s (SeparationQuotient.mk a))
      ⊢ Disjoint (nhdsSet s) (nhds (SeparationQuotient.mk a))
    -/
    rw [← disjoint_comap_iff surjective_mk, comap_mk_nhds_mk, comap_mk_nhdsSet]
    /-
      case intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : RegularSpace X
      s : Set (SeparationQuotient X)
      hs : IsClosed s
      a : X
      ha : Not (Membership.mem s (SeparationQuotient.mk a))
      ⊢ Disjoint (nhdsSet (Set.preimage SeparationQuotient.mk s)) (nhds a)
    -/
    exact RegularSpace.regular (hs.preimage continuous_mk) ha
    /-
      🎉 no goals
    -/


/-- A topological space is said to be a *normal space* if any two disjoint closed sets
have disjoint open neighborhoods. -/
class NormalSpace (X : Type u) [TopologicalSpace X] : Prop where
  /-- Two disjoint sets in a normal space admit disjoint neighbourhoods. -/
  normal : ∀ s t : Set X, IsClosed s → IsClosed t → Disjoint s t → SeparatedNhds s t


theorem normal_separation [NormalSpace X] {s t : Set X} (H1 : IsClosed s) (H2 : IsClosed t)
    (H3 : Disjoint s t) : SeparatedNhds s t :=
  NormalSpace.normal s t H1 H2 H3


theorem disjoint_nhdsSet_nhdsSet [NormalSpace X] {s t : Set X} (hs : IsClosed s) (ht : IsClosed t)
    (hd : Disjoint s t) : Disjoint (𝓝ˢ s) (𝓝ˢ t) :=
  (normal_separation hs ht hd).disjoint_nhdsSet


theorem normal_exists_closure_subset [NormalSpace X] {s t : Set X} (hs : IsClosed s) (ht : IsOpen t)
    (hst : s ⊆ t) : ∃ u, IsOpen u ∧ s ⊆ u ∧ closure u ⊆ t := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : NormalSpace X
    s t : Set X
    hs : IsClosed s
    ht : IsOpen t
    hst : HasSubset.Subset s t
    ⊢ Exists fun u => And (IsOpen u) (And (HasSubset.Subset s u) (HasSubset.Subset …
  -/
  have : Disjoint s tᶜ := Set.disjoint_left.mpr fun x hxs hxt => hxt (hst hxs)
  rcases normal_separation hs (isClosed_compl_iff.2 ht) this with
    ⟨s', t', hs', ht', hss', htt', hs't'⟩
  refine ⟨s', hs', hss', Subset.trans (closure_minimal ?_ (isClosed_compl_iff.2 ht'))
    (compl_subset_comm.1 htt')⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : NormalSpace X
    s t : Set X
    hs : IsClosed s
    ht : IsOpen t
    hst : HasSubset.Subset s t
    this : Disjoint s (HasCompl.compl t)
    s' t' : Set X
    hs' : IsOpen s'
    ht' : IsOpen t'
    hss' : HasSubset.Subset s s'
    htt' : HasSubset.Subset (HasCompl.compl t) t'
    hs't' : Disjoint s' t'
    ⊢ HasSubset.Subset s' (HasCompl.compl t')
  -/
  exact fun x hxs hxt => hs't'.le_bot ⟨hxs, hxt⟩
  /-
    🎉 no goals
  -/


/-- If the codomain of a closed embedding is a normal space, then so is the domain. -/
protected theorem Topology.IsClosedEmbedding.normalSpace [TopologicalSpace Y] [NormalSpace Y]
    {f : X → Y} (hf : IsClosedEmbedding f) : NormalSpace X where
  normal s t hs ht hst := by
    have H : SeparatedNhds (f '' s) (f '' t) :=
      NormalSpace.normal (f '' s) (f '' t) (hf.isClosedMap s hs) (hf.isClosedMap t ht)
        (disjoint_image_of_injective hf.injective hst)
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : X → Y
      hf : Topology.IsClosedEmbedding f
      s t : Set X
      hs : IsClosed s
      ht : IsClosed t
      hst : Disjoint s t
      H : SeparatedNhds (Set.image f s) (Set.image f t)
      ⊢ SeparatedNhds s t
    -/
    exact (H.preimage hf.continuous).mono (subset_preimage_image _ _) (subset_preimage_image _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.normalSpace := IsClosedEmbedding.normalSpace


instance (priority := 100) NormalSpace.of_compactSpace_r1Space [CompactSpace X] [R1Space X] :
    NormalSpace X where
  normal _s _t hs ht := .of_isCompact_isCompact_isClosed hs.isCompact ht.isCompact ht


set_option pp.universes true in
/-- A regular topological space with a Lindelöf topology is a normal space. A consequence of e.g.
Corollaries 20.8 and 20.10 of [Willard's *General Topology*][zbMATH02107988] (without the
assumption of Hausdorff). -/
instance (priority := 100) NormalSpace.of_regularSpace_lindelofSpace
    [RegularSpace X] [LindelofSpace X] : NormalSpace X where
  normal _ _ hcl kcl hkdis :=
    hasSeparatingCovers_iff_separatedNhds.mp
    ⟨hcl.HasSeparatingCover kcl hkdis, kcl.HasSeparatingCover hcl (Disjoint.symm hkdis)⟩


instance (priority := 100) NormalSpace.of_regularSpace_secondCountableTopology
    [RegularSpace X] [SecondCountableTopology X] : NormalSpace X :=
  of_regularSpace_lindelofSpace


/-- A T₄ space is a normal T₁ space. -/
class T4Space (X : Type u) [TopologicalSpace X] extends T1Space X, NormalSpace X : Prop


instance (priority := 100) [T1Space X] [NormalSpace X] : T4Space X := ⟨⟩

-- see Note [lower instance priority]

instance (priority := 100) T4Space.t3Space [T4Space X] : T3Space X where
  regular hs hxs := by simpa only [nhdsSet_singleton] using (normal_separation hs isClosed_singleton
    (disjoint_singleton_right.mpr hxs)).disjoint_nhdsSet


/-- If the codomain of a closed embedding is a T₄ space, then so is the domain. -/
protected theorem Topology.IsClosedEmbedding.t4Space [TopologicalSpace Y] [T4Space Y] {f : X → Y}
    (hf : IsClosedEmbedding f) : T4Space X where
  toT1Space := hf.isEmbedding.t1Space
  toNormalSpace := hf.normalSpace


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.t4Space := IsClosedEmbedding.t4Space


instance ULift.instT4Space [T4Space X] : T4Space (ULift X) := IsClosedEmbedding.uliftDown.t4Space


/-- The `SeparationQuotient` of a normal space is a normal space. -/
instance [NormalSpace X] : NormalSpace (SeparationQuotient X) where
  normal s t hs ht hd := separatedNhds_iff_disjoint.2 <| by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : NormalSpace X
      s t : Set (SeparationQuotient X)
      hs : IsClosed s
      ht : IsClosed t
      hd : Disjoint s t
      ⊢ Disjoint (nhdsSet s) (nhdsSet t)
    -/
    rw [← disjoint_comap_iff surjective_mk, comap_mk_nhdsSet, comap_mk_nhdsSet]
    exact disjoint_nhdsSet_nhdsSet (hs.preimage continuous_mk) (ht.preimage continuous_mk)
      (hd.preimage mk)


/-- A topological space `X` is a *completely normal space* provided that for any two sets `s`, `t`
such that if both `closure s` is disjoint with `t`, and `s` is disjoint with `closure t`,
then there exist disjoint neighbourhoods of `s` and `t`. -/
class CompletelyNormalSpace (X : Type u) [TopologicalSpace X] : Prop where
  /-- If `closure s` is disjoint with `t`, and `s` is disjoint with `closure t`, then `s` and `t`
  admit disjoint neighbourhoods. -/
  completely_normal :
    ∀ ⦃s t : Set X⦄, Disjoint (closure s) t → Disjoint s (closure t) → Disjoint (𝓝ˢ s) (𝓝ˢ t)


/-- A completely normal space is a normal space. -/
instance (priority := 100) CompletelyNormalSpace.toNormalSpace
    [CompletelyNormalSpace X] : NormalSpace X where
  normal s t hs ht hd := separatedNhds_iff_disjoint.2 <|
                          /-
                            X : Type u_1
                            Y : Type u_2
                            inst✝¹ : TopologicalSpace X
                            inst✝ : CompletelyNormalSpace X
                            s t : Set X
                            hs : IsClosed s
                            ht : IsClosed t
                            hd : Disjoint s t
                            ⊢ Disjoint (closure s) t
                          -/
                          /-
                            🎉 no goals
                          -/
    completely_normal (by rwa [hs.closure_eq]) (by rwa [ht.closure_eq])
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Topology.IsEmbedding.completelyNormalSpace [TopologicalSpace Y] [CompletelyNormalSpace Y]
    {e : X → Y} (he : IsEmbedding e) : CompletelyNormalSpace X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompletelyNormalSpace Y
    e : X → Y
    he : Topology.IsEmbedding e
    ⊢ CompletelyNormalSpace X
  -/
  refine ⟨fun s t hd₁ hd₂ => ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompletelyNormalSpace Y
    e : X → Y
    he : Topology.IsEmbedding e
    s t : Set X
    hd₁ : Disjoint (closure s) t
    hd₂ : Disjoint s (closure t)
    ⊢ Disjoint (nhdsSet s) (nhdsSet t)
  -/
  simp only [he.isInducing.nhdsSet_eq_comap]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompletelyNormalSpace Y
    e : X → Y
    he : Topology.IsEmbedding e
    s t : Set X
    hd₁ : Disjoint (closure s) t
    hd₂ : Disjoint s (closure t)
    ⊢ Disjoint (Filter.comap e (nhdsSet (Set.image e s))) (Filter.comap e (nhdsSet …
  -/
  refine disjoint_comap (completely_normal ?_ ?_)
  · rwa [← subset_compl_iff_disjoint_left, image_subset_iff, preimage_compl,
      ← he.closure_eq_preimage_closure_image, subset_compl_iff_disjoint_left]
  · rwa [← subset_compl_iff_disjoint_right, image_subset_iff, preimage_compl,
      ← he.closure_eq_preimage_closure_image, subset_compl_iff_disjoint_right]


@[deprecated (since := "2024-10-26")]
alias Embedding.completelyNormalSpace := IsEmbedding.completelyNormalSpace


/-- A subspace of a completely normal space is a completely normal space. -/
instance [CompletelyNormalSpace X] {p : X → Prop} : CompletelyNormalSpace { x // p x } :=
  IsEmbedding.subtypeVal.completelyNormalSpace


instance ULift.instCompletelyNormalSpace [CompletelyNormalSpace X] :
    CompletelyNormalSpace (ULift X) :=
  IsEmbedding.uliftDown.completelyNormalSpace


/-- A T₅ space is a completely normal T₁ space. -/
class T5Space (X : Type u) [TopologicalSpace X] extends T1Space X, CompletelyNormalSpace X : Prop


theorem Topology.IsEmbedding.t5Space [TopologicalSpace Y] [T5Space Y] {e : X → Y}
    (he : IsEmbedding e) : T5Space X where
  __ := he.t1Space
  completely_normal := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : T5Space Y
      e : X → Y
      he : Topology.IsEmbedding e
      ⊢ ∀ ⦃s t : Set X⦄, Disjoint (closure s) t → Disjoint s (closure t) → Disjoint  …
    -/
    have := he.completelyNormalSpace
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : T5Space Y
      e : X → Y
      he : Topology.IsEmbedding e
      this : CompletelyNormalSpace X
      ⊢ ∀ ⦃s t : Set X⦄, Disjoint (closure s) t → Disjoint s (closure t) → Disjoint  …
    -/
    exact completely_normal
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias Embedding.t5Space := IsEmbedding.t5Space

-- see Note [lower instance priority]

/-- A `T₅` space is a `T₄` space. -/
instance (priority := 100) T5Space.toT4Space [T5Space X] : T4Space X where
  -- follows from type-class inference


/-- A subspace of a T₅ space is a T₅ space. -/
instance [T5Space X] {p : X → Prop} : T5Space { x // p x } :=
  IsEmbedding.subtypeVal.t5Space


instance ULift.instT5Space [T5Space X] : T5Space (ULift X) :=
  IsEmbedding.uliftDown.t5Space


/-- The `SeparationQuotient` of a completely normal R₀ space is a T₅ space. -/
instance [CompletelyNormalSpace X] [R0Space X] : T5Space (SeparationQuotient X) where
  t1 := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : CompletelyNormalSpace X
      inst✝ : R0Space X
      ⊢ ∀ (x : SeparationQuotient X), IsClosed (Singleton.singleton x)
    -/
    rwa [((t1Space_TFAE (SeparationQuotient X)).out 1 0 :), SeparationQuotient.t1Space_iff]
    /-
      🎉 no goals
    -/
  completely_normal s t hd₁ hd₂ := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : CompletelyNormalSpace X
      inst✝ : R0Space X
      s t : Set (SeparationQuotient X)
      hd₁ : Disjoint (closure s) t
      hd₂ : Disjoint s (closure t)
      ⊢ Disjoint (nhdsSet s) (nhdsSet t)
    -/
    rw [← disjoint_comap_iff surjective_mk, comap_mk_nhdsSet, comap_mk_nhdsSet]
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : CompletelyNormalSpace X
      inst✝ : R0Space X
      s t : Set (SeparationQuotient X)
      hd₁ : Disjoint (closure s) t
      hd₂ : Disjoint s (closure t)
      ⊢ Disjoint (nhdsSet (Set.preimage SeparationQuotient.mk s)) (nhdsSet (Set.prei …
    -/
    apply completely_normal <;> rw [← preimage_mk_closure]
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : CompletelyNormalSpace X
      inst✝ : R0Space X
      s t : Set (SeparationQuotient X)
      hd₁ : Disjoint (closure s) t
      hd₂ : Disjoint s (closure t)
      ⊢ Disjoint (Set.preimage SeparationQuotient.mk (closure s)) (Set.preimage Sepa …
    -/
    exacts [hd₁.preimage mk, hd₂.preimage mk]
    /-
      🎉 no goals
    -/


/-- In a compact T₂ space, the connected component of a point equals the intersection of all
its clopen neighbourhoods. -/
theorem connectedComponent_eq_iInter_isClopen [T2Space X] [CompactSpace X] (x : X) :
    connectedComponent x = ⋂ s : { s : Set X // IsClopen s ∧ x ∈ s }, s := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    x : X
    ⊢ Eq (connectedComponent x) (Set.iInter fun s => ↑s)
  -/
  apply Subset.antisymm connectedComponent_subset_iInter_isClopen
  -- Reduce to showing that the clopen intersection is connected.
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    x : X
    ⊢ HasSubset.Subset (Set.iInter fun Z => ↑Z) (connectedComponent x)
  -/
  refine IsPreconnected.subset_connectedComponent ?_ (mem_iInter.2 fun s => s.2.2)
  -- We do this by showing that any disjoint cover by two closed sets implies
  -- that one of these closed sets must contain our whole thing.
  -- To reduce to the case where the cover is disjoint on all of `X` we need that `s` is closed
  have hs : @IsClosed X _ (⋂ s : { s : Set X // IsClopen s ∧ x ∈ s }, s) :=
    isClosed_iInter fun s => s.2.1.1
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    x : X
    hs : IsClosed (Set.iInter fun s => ↑s)
    ⊢ IsPreconnected (Set.iInter fun Z => ↑Z)
  -/
  rw [isPreconnected_iff_subset_of_fully_disjoint_closed hs]
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    x : X
    hs : IsClosed (Set.iInter fun s => ↑s)
    ⊢ ∀ (u v : Set X), IsClosed u → IsClosed v → HasSubset.Subset (Set.iInter fun  …
  -/
  intro a b ha hb hab ab_disj
  -- Since our space is normal, we get two larger disjoint open sets containing the disjoint
  -- closed sets. If we can show that our intersection is a subset of any of these we can then
  -- "descend" this to show that it is a subset of either a or b.
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    x : X
    hs : IsClosed (Set.iInter fun s => ↑s)
    a b : Set X
    ha : IsClosed a
    hb : IsClosed b
    hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
    ab_disj : Disjoint a b
    ⊢ Or (HasSubset.Subset (Set.iInter fun s => ↑s) a) (HasSubset.Subset (Set.iInt …
  -/
  rcases normal_separation ha hb ab_disj with ⟨u, v, hu, hv, hau, hbv, huv⟩
  obtain ⟨s, H⟩ : ∃ s : Set X, IsClopen s ∧ x ∈ s ∧ s ⊆ u ∪ v := by
    /- Now we find a clopen set `s` around `x`, contained in `u ∪ v`. We utilize the fact that
    `X \ u ∪ v` will be compact, so there must be some finite intersection of clopen neighbourhoods
    of `X` disjoint to it, but a finite intersection of clopen sets is clopen,
    so we let this be our `s`. -/
    have H1 := (hu.union hv).isClosed_compl.isCompact.inter_iInter_nonempty
      (fun s : { s : Set X // IsClopen s ∧ x ∈ s } => s) fun s => s.2.1.1
    rw [← not_disjoint_iff_nonempty_inter, imp_not_comm, not_forall] at H1
    cases' H1 (disjoint_compl_left_iff_subset.2 <| hab.trans <| union_subset_union hau hbv)
      with si H2
    refine ⟨⋂ U ∈ si, Subtype.val U, ?_, ?_, ?_⟩
    · exact isClopen_biInter_finset fun s _ => s.2.1
    · exact mem_iInter₂.2 fun s _ => s.2.2
    · rwa [← disjoint_compl_left_iff_subset, disjoint_iff_inter_eq_empty,
        ← not_nonempty_iff_eq_empty]
  -- So, we get a disjoint decomposition `s = s ∩ u ∪ s ∩ v` of clopen sets. The intersection of all
  -- clopen neighbourhoods will then lie in whichever of u or v x lies in and hence will be a subset
  -- of either a or b.
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : CompactSpace X
      x : X
      hs : IsClosed (Set.iInter fun s => ↑s)
      a b : Set X
      ha : IsClosed a
      hb : IsClosed b
      hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
      ab_disj : Disjoint a b
      u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      hau : HasSubset.Subset a u
      hbv : HasSubset.Subset b v
      huv : Disjoint u v
      s : Set X
      H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
      ⊢ Or (HasSubset.Subset (Set.iInter fun s => ↑s) a) (HasSubset.Subset (Set.iInt …
    -/
  · have H1 := isClopen_inter_of_disjoint_cover_clopen H.1 H.2.2 hu hv huv
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : CompactSpace X
      x : X
      hs : IsClosed (Set.iInter fun s => ↑s)
      a b : Set X
      ha : IsClosed a
      hb : IsClosed b
      hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
      ab_disj : Disjoint a b
      u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      hau : HasSubset.Subset a u
      hbv : HasSubset.Subset b v
      huv : Disjoint u v
      s : Set X
      H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
      H1 : IsClopen (Inter.inter s u)
      ⊢ Or (HasSubset.Subset (Set.iInter fun s => ↑s) a) (HasSubset.Subset (Set.iInt …
    -/
    rw [union_comm] at H
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : CompactSpace X
      x : X
      hs : IsClosed (Set.iInter fun s => ↑s)
      a b : Set X
      ha : IsClosed a
      hb : IsClosed b
      hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
      ab_disj : Disjoint a b
      u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      hau : HasSubset.Subset a u
      hbv : HasSubset.Subset b v
      huv : Disjoint u v
      s : Set X
      H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
      H1 : IsClopen (Inter.inter s u)
      ⊢ Or (HasSubset.Subset (Set.iInter fun s => ↑s) a) (HasSubset.Subset (Set.iInt …
    -/
    have H2 := isClopen_inter_of_disjoint_cover_clopen H.1 H.2.2 hv hu huv.symm
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : CompactSpace X
      x : X
      hs : IsClosed (Set.iInter fun s => ↑s)
      a b : Set X
      ha : IsClosed a
      hb : IsClosed b
      hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
      ab_disj : Disjoint a b
      u v : Set X
      hu : IsOpen u
      hv : IsOpen v
      hau : HasSubset.Subset a u
      hbv : HasSubset.Subset b v
      huv : Disjoint u v
      s : Set X
      H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
      H1 : IsClopen (Inter.inter s u)
      H2 : IsClopen (Inter.inter s v)
      ⊢ Or (HasSubset.Subset (Set.iInter fun s => ↑s) a) (HasSubset.Subset (Set.iInt …
    -/
    by_cases hxu : x ∈ u <;> [left; right]
    -- The x ∈ u case.
    · suffices ⋂ s : { s : Set X // IsClopen s ∧ x ∈ s }, ↑s ⊆ u
        from Disjoint.left_le_of_le_sup_right hab (huv.mono this hbv)
        /-
          case pos.h
          X : Type u_1
          inst✝² : TopologicalSpace X
          inst✝¹ : T2Space X
          inst✝ : CompactSpace X
          x : X
          hs : IsClosed (Set.iInter fun s => ↑s)
          a b : Set X
          ha : IsClosed a
          hb : IsClosed b
          hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
          ab_disj : Disjoint a b
          u v : Set X
          hu : IsOpen u
          hv : IsOpen v
          hau : HasSubset.Subset a u
          hbv : HasSubset.Subset b v
          huv : Disjoint u v
          s : Set X
          H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
          H1 : IsClopen (Inter.inter s u)
          H2 : IsClopen (Inter.inter s v)
          hxu : Membership.mem u x
          ⊢ HasSubset.Subset (Set.iInter fun s => ↑s) u
        -/
      · apply Subset.trans _ s.inter_subset_right
        exact iInter_subset (fun s : { s : Set X // IsClopen s ∧ x ∈ s } => s.1)
          ⟨s ∩ u, H1, mem_inter H.2.1 hxu⟩
    -- If x ∉ u, we get x ∈ v since x ∈ u ∪ v. The rest is then like the x ∈ u case.
    · have h1 : x ∈ v :=
        (hab.trans (union_subset_union hau hbv) (mem_iInter.2 fun i => i.2.2)).resolve_left hxu
      suffices ⋂ s : { s : Set X // IsClopen s ∧ x ∈ s }, ↑s ⊆ v
        from (huv.symm.mono this hau).left_le_of_le_sup_left hab
        /-
          case neg.h
          X : Type u_1
          inst✝² : TopologicalSpace X
          inst✝¹ : T2Space X
          inst✝ : CompactSpace X
          x : X
          hs : IsClosed (Set.iInter fun s => ↑s)
          a b : Set X
          ha : IsClosed a
          hb : IsClosed b
          hab : HasSubset.Subset (Set.iInter fun s => ↑s) (Union.union a b)
          ab_disj : Disjoint a b
          u v : Set X
          hu : IsOpen u
          hv : IsOpen v
          hau : HasSubset.Subset a u
          hbv : HasSubset.Subset b v
          huv : Disjoint u v
          s : Set X
          H : And (IsClopen s) (And (Membership.mem s x) (HasSubset.Subset s (Union.unio …
          H1 : IsClopen (Inter.inter s u)
          H2 : IsClopen (Inter.inter s v)
          hxu : Not (Membership.mem u x)
          h1 : Membership.mem v x
          ⊢ HasSubset.Subset (Set.iInter fun s => ↑s) v
        -/
      · refine Subset.trans ?_ s.inter_subset_right
        exact iInter_subset (fun s : { s : Set X // IsClopen s ∧ x ∈ s } => s.1)
          ⟨s ∩ v, H2, mem_inter H.2.1 h1⟩


/-- `ConnectedComponents X` is Hausdorff when `X` is Hausdorff and compact -/
instance ConnectedComponents.t2 [T2Space X] [CompactSpace X] : T2Space (ConnectedComponents X) := by
  -- Proof follows that of: https://stacks.math.columbia.edu/tag/0900
  -- Fix 2 distinct connected components, with points a and b
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    ⊢ T2Space (ConnectedComponents X)
  -/
  refine ⟨ConnectedComponents.surjective_coe.forall₂.2 fun a b ne => ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (ConnectedComponents.mk a) (ConnectedComponents.mk b)
    ⊢ (fun x y => Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) ( …
  -/
  rw [ConnectedComponents.coe_ne_coe] at ne
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (connectedComponent a) (connectedComponent b)
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  have h := connectedComponent_disjoint ne
  -- write ↑b as the intersection of all clopen subsets containing it
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (connectedComponent a) (connectedComponent b)
    h : Disjoint (connectedComponent a) (connectedComponent b)
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  rw [connectedComponent_eq_iInter_isClopen b, disjoint_iff_inter_eq_empty] at h
  -- Now we show that this can be reduced to some clopen containing `↑b` being disjoint to `↑a`
  obtain ⟨U, V, hU, ha, hb, rfl⟩ : ∃ (U : Set X) (V : Set (ConnectedComponents X)),
      IsClopen U ∧ connectedComponent a ∩ U = ∅ ∧ connectedComponent b ⊆ U ∧ (↑) ⁻¹' V = U := by
    have h :=
      (isClosed_connectedComponent (α := X)).isCompact.elim_finite_subfamily_closed
        _ (fun s : { s : Set X // IsClopen s ∧ b ∈ s } => s.2.1.1) h
    cases' h with fin_a ha
    -- This clopen and its complement will separate the connected components of `a` and `b`
    set U : Set X := ⋂ (i : { s // IsClopen s ∧ b ∈ s }) (_ : i ∈ fin_a), i
    have hU : IsClopen U := isClopen_biInter_finset fun i _ => i.2.1
    exact ⟨U, (↑) '' U, hU, ha, subset_iInter₂ fun s _ => s.2.1.connectedComponent_subset s.2.2,
      (connectedComponents_preimage_image U).symm ▸ hU.biUnion_connectedComponent_eq⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (connectedComponent a) (connectedComponent b)
    h : Eq (Inter.inter (connectedComponent a) (Set.iInter fun s => ↑s)) EmptyColl …
    V : Set (ConnectedComponents X)
    hU : IsClopen (Set.preimage ConnectedComponents.mk V)
    ha : Eq (Inter.inter (connectedComponent a) (Set.preimage ConnectedComponents. …
    hb : HasSubset.Subset (connectedComponent b) (Set.preimage ConnectedComponents …
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  rw [ConnectedComponents.isQuotientMap_coe.isClopen_preimage] at hU
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (connectedComponent a) (connectedComponent b)
    h : Eq (Inter.inter (connectedComponent a) (Set.iInter fun s => ↑s)) EmptyColl …
    V : Set (ConnectedComponents X)
    hU : IsClopen V
    ha : Eq (Inter.inter (connectedComponent a) (Set.preimage ConnectedComponents. …
    hb : HasSubset.Subset (connectedComponent b) (Set.preimage ConnectedComponents …
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  refine ⟨Vᶜ, V, hU.compl.isOpen, hU.isOpen, ?_, hb mem_connectedComponent, disjoint_compl_left⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    a b : X
    ne : Ne (connectedComponent a) (connectedComponent b)
    h : Eq (Inter.inter (connectedComponent a) (Set.iInter fun s => ↑s)) EmptyColl …
    V : Set (ConnectedComponents X)
    hU : IsClopen V
    ha : Eq (Inter.inter (connectedComponent a) (Set.preimage ConnectedComponents. …
    hb : HasSubset.Subset (connectedComponent b) (Set.preimage ConnectedComponents …
    ⊢ Membership.mem (HasCompl.compl V) (ConnectedComponents.mk a)
  -/
  exact fun h => flip Set.Nonempty.ne_empty ha ⟨a, mem_connectedComponent, h⟩
  /-
    🎉 no goals
  -/

