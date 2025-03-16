@[simp]
theorem ker_lsingle (a : α) : ker (lsingle a : M →ₗ[R] α →₀ M) = ⊥ :=
  ker_eq_bot_of_injective (single_injective a)


theorem lsingle_range_le_ker_lapply (s t : Set α) (h : Disjoint s t) :
    ⨆ a ∈ s, LinearMap.range (lsingle a : M →ₗ[R] α →₀ M) ≤
      ⨅ a ∈ t, ker (lapply a : (α →₀ M) →ₗ[R] M) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    h : Disjoint s t
    ⊢ LE.le (iSup fun a => iSup fun h => LinearMap.range (Finsupp.lsingle a)) (iIn …
  -/
  refine iSup_le fun a₁ => iSup_le fun h₁ => range_le_iff_comap.2 ?_
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    h : Disjoint s t
    a₁ : α
    h₁ : Membership.mem s a₁
    ⊢ Eq (Submodule.comap (Finsupp.lsingle a₁) (iInf fun a => iInf fun h => Linear …
  -/
  simp only [(ker_comp _ _).symm, eq_top_iff, SetLike.le_def, mem_ker, comap_iInf, mem_iInf]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    h : Disjoint s t
    a₁ : α
    h₁ : Membership.mem s a₁
    ⊢ ∀ ⦃x : M⦄, Membership.mem Top.top x → ∀ (i : α), Membership.mem t i → Eq ((( …
  -/
  intro b _ a₂ h₂
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    h : Disjoint s t
    a₁ : α
    h₁ : Membership.mem s a₁
    b : M
    a✝ : Membership.mem Top.top b
    a₂ : α
    h₂ : Membership.mem t a₂
    ⊢ Eq (((Finsupp.lapply a₂).comp (Finsupp.lsingle a₁)) b) 0
  -/
  have : a₁ ≠ a₂ := fun eq => h.le_bot ⟨h₁, eq.symm ▸ h₂⟩
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    h : Disjoint s t
    a₁ : α
    h₁ : Membership.mem s a₁
    b : M
    a✝ : Membership.mem Top.top b
    a₂ : α
    h₂ : Membership.mem t a₂
    this : Ne a₁ a₂
    ⊢ Eq (((Finsupp.lapply a₂).comp (Finsupp.lsingle a₁)) b) 0
  -/
  exact single_eq_of_ne this
  /-
    🎉 no goals
  -/


theorem iInf_ker_lapply_le_bot : ⨅ a, ker (lapply a : (α →₀ M) →ₗ[R] M) ≤ ⊥ := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ LE.le (iInf fun a => LinearMap.ker (Finsupp.lapply a)) Bot.bot
  -/
  simp only [SetLike.le_def, mem_iInf, mem_ker, mem_bot, lapply_apply]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ ∀ ⦃x : Finsupp α M⦄, (∀ (i : α), Eq (x i) 0) → Eq x 0
  -/
  exact fun a h => Finsupp.ext h
  /-
    🎉 no goals
  -/


theorem iSup_lsingle_range : ⨆ a, LinearMap.range (lsingle a : M →ₗ[R] α →₀ M) = ⊤ := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (iSup fun a => LinearMap.range (Finsupp.lsingle a)) Top.top
  -/
  refine eq_top_iff.2 <| SetLike.le_def.2 fun f _ => ?_
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Finsupp α M
    x✝ : Membership.mem Top.top f
    ⊢ Membership.mem (iSup fun a => LinearMap.range (Finsupp.lsingle a)) f
  -/
  rw [← sum_single f]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Finsupp α M
    x✝ : Membership.mem Top.top f
    ⊢ Membership.mem (iSup fun a => LinearMap.range (Finsupp.lsingle a)) (f.sum Fi …
  -/
  exact sum_mem fun a _ => Submodule.mem_iSup_of_mem a ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


theorem disjoint_lsingle_lsingle (s t : Set α) (hs : Disjoint s t) :
    Disjoint (⨆ a ∈ s, LinearMap.range (lsingle a : M →ₗ[R] α →₀ M))
      (⨆ a ∈ t, LinearMap.range (lsingle a : M →ₗ[R] α →₀ M)) := by
  -- Porting note: 2 placeholders are added to prevent timeout.
  refine
    (Disjoint.mono
      (lsingle_range_le_ker_lapply s sᶜ ?_)
      (lsingle_range_le_ker_lapply t tᶜ ?_))
      ?_
    /-
      case refine_1
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s t : Set α
      hs : Disjoint s t
      ⊢ Disjoint s (HasCompl.compl s)
    -/
  · apply disjoint_compl_right
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s t : Set α
      hs : Disjoint s t
      ⊢ Disjoint t (HasCompl.compl t)
    -/
  · apply disjoint_compl_right
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    hs : Disjoint s t
    ⊢ Disjoint (iInf fun a => iInf fun h => LinearMap.ker (Finsupp.lapply a)) (iIn …
  -/
  rw [disjoint_iff_inf_le]
  /-
    case refine_3
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    hs : Disjoint s t
    ⊢ LE.le (Min.min (iInf fun a => iInf fun h => LinearMap.ker (Finsupp.lapply a) …
  -/
  refine le_trans (le_iInf fun i => ?_) iInf_ker_lapply_le_bot
  classical
    by_cases his : i ∈ s
    · by_cases hit : i ∈ t
      · exact (hs.le_bot ⟨his, hit⟩).elim
      exact inf_le_of_right_le (iInf_le_of_le i <| iInf_le _ hit)
    exact inf_le_of_left_le (iInf_le_of_le i <| iInf_le _ his)


theorem span_single_image (s : Set M) (a : α) :
    Submodule.span R (single a '' s) = (Submodule.span R s).map (lsingle a : M →ₗ[R] α →₀ M) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    a : α
    ⊢ Eq (Submodule.span R (Set.image (Finsupp.single a) s)) (Submodule.map (Finsu …
  -/
  rw [← span_image]; rfl
                     /-
                       🎉 no goals
                     -/


theorem Submodule.exists_finset_of_mem_iSup {ι : Sort _} (p : ι → Submodule R M) {m : M}
    (hm : m ∈ ⨆ i, p i) : ∃ s : Finset ι, m ∈ ⨆ i ∈ s, p i := by
  have :=
    CompleteLattice.IsCompactElement.exists_finset_of_le_iSup (Submodule R M)
      (Submodule.singleton_span_isCompactElement m) p
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_4
    p : ι → Submodule R M
    m : M
    hm : Membership.mem (iSup fun i => p i) m
    this : LE.le (Submodule.span R (Singleton.singleton m)) (iSup fun i => p i) →  …
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => p i) m
  -/
  simp only [Submodule.span_singleton_le_iff_mem] at this
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_4
    p : ι → Submodule R M
    m : M
    hm : Membership.mem (iSup fun i => p i) m
    this : Membership.mem (iSup fun i => p i) m → Exists fun s => Membership.mem ( …
    ⊢ Exists fun s => Membership.mem (iSup fun i => iSup fun h => p i) m
  -/
  exact this hm
  /-
    🎉 no goals
  -/


/-- `Submodule.exists_finset_of_mem_iSup` as an `iff` -/
theorem Submodule.mem_iSup_iff_exists_finset {ι : Sort _} {p : ι → Submodule R M} {m : M} :
    (m ∈ ⨆ i, p i) ↔ ∃ s : Finset ι, m ∈ ⨆ i ∈ s, p i :=
  ⟨Submodule.exists_finset_of_mem_iSup p, fun ⟨_, hs⟩ =>
    iSup_mono (fun i => (iSup_const_le : _ ≤ p i)) hs⟩


theorem Submodule.mem_sSup_iff_exists_finset {S : Set (Submodule R M)} {m : M} :
    m ∈ sSup S ↔ ∃ s : Finset (Submodule R M), ↑s ⊆ S ∧ m ∈ ⨆ i ∈ s, i := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Set (Submodule R M)
    m : M
    ⊢ Iff (Membership.mem (SupSet.sSup S) m) (Exists fun s => And (HasSubset.Subse …
  -/
  rw [sSup_eq_iSup, iSup_subtype', Submodule.mem_iSup_iff_exists_finset]
  refine ⟨fun ⟨s, hs⟩ ↦ ⟨s.map (Function.Embedding.subtype S), ?_, ?_⟩,
          fun ⟨s, hsS, hs⟩ ↦ ⟨s.preimage (↑) Subtype.coe_injective.injOn, ?_⟩⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Set (Submodule R M)
      m : M
      x✝ : Exists fun s => Membership.mem (iSup fun i => iSup fun h => ↑i) m
      s : Finset (Subtype (Membership.mem S))
      hs : Membership.mem (iSup fun i => iSup fun h => ↑i) m
      ⊢ HasSubset.Subset (↑(Finset.map (Function.Embedding.subtype S) s)) S
    -/
  · simpa using fun x _ ↦ x.property
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Set (Submodule R M)
      m : M
      x✝ : Exists fun s => Membership.mem (iSup fun i => iSup fun h => ↑i) m
      s : Finset (Subtype (Membership.mem S))
      hs : Membership.mem (iSup fun i => iSup fun h => ↑i) m
      ⊢ Membership.mem (iSup fun i => iSup fun h => i) m
    -/
  · suffices m ∈ ⨆ (i) (hi : i ∈ S) (_ : ⟨i, hi⟩ ∈ s), i by simpa
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Set (Submodule R M)
      m : M
      x✝ : Exists fun s => Membership.mem (iSup fun i => iSup fun h => ↑i) m
      s : Finset (Subtype (Membership.mem S))
      hs : Membership.mem (iSup fun i => iSup fun h => ↑i) m
      ⊢ Membership.mem (iSup fun i => iSup fun hi => iSup fun x => i) m
    -/
    rwa [iSup_subtype']
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Set (Submodule R M)
      m : M
      x✝ : Exists fun s => And (HasSubset.Subset (↑s) S) (Membership.mem (iSup fun i …
      s : Finset (Submodule R M)
      hsS : HasSubset.Subset (↑s) S
      hs : Membership.mem (iSup fun i => iSup fun h => i) m
      ⊢ Membership.mem (iSup fun i => iSup fun h => ↑i) m
    -/
  · have : ⨆ (i) (_ : i ∈ S ∧ i ∈ s), i = ⨆ (i) (_ : i ∈ s), i := by convert rfl; aesop
    /-
      case refine_3
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      S : Set (Submodule R M)
      m : M
      x✝ : Exists fun s => And (HasSubset.Subset (↑s) S) (Membership.mem (iSup fun i …
      s : Finset (Submodule R M)
      hsS : HasSubset.Subset (↑s) S
      hs : Membership.mem (iSup fun i => iSup fun h => i) m
      this : Eq (iSup fun i => iSup fun x => i) (iSup fun i => iSup fun x => i)
      ⊢ Membership.mem (iSup fun i => iSup fun h => ↑i) m
    -/
    simpa only [Finset.mem_preimage, iSup_subtype, iSup_and', this]
    /-
      🎉 no goals
    -/

