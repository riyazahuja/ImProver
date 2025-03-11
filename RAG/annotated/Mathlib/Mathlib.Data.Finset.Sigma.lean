/-- `s.sigma t` is the finset of dependent pairs `⟨i, a⟩` such that `i ∈ s` and `a ∈ t i`. -/
protected def sigma : Finset (Σi, α i) :=
  ⟨_, s.nodup.sigma fun i => (t i).nodup⟩


@[simp]
theorem mem_sigma {a : Σi, α i} : a ∈ s.sigma t ↔ a.1 ∈ s ∧ a.2 ∈ t a.1 :=
  Multiset.mem_sigma


@[simp, norm_cast]
theorem coe_sigma (s : Finset ι) (t : ∀ i, Finset (α i)) :
    (s.sigma t : Set (Σ i, α i)) = (s : Set ι).sigma fun i ↦ (t i : Set (α i)) :=
  Set.ext fun _ => mem_sigma


@[simp]
                                                                              /-
                                                                                ι : Type u_1
                                                                                α : ι → Type u_2
                                                                                s : Finset ι
                                                                                t : (i : ι) → Finset (α i)
                                                                                ⊢ Iff (s.sigma t).Nonempty (Exists fun i => And (Membership.mem s i) (t i).Non …
                                                                              -/
theorem sigma_nonempty : (s.sigma t).Nonempty ↔ ∃ i ∈ s, (t i).Nonempty := by simp [Finset.Nonempty]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.sigma_nonempty_of_exists_nonempty⟩ := sigma_nonempty


@[simp]
theorem sigma_eq_empty : s.sigma t = ∅ ↔ ∀ i ∈ s, t i = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    ⊢ Iff (Eq (s.sigma t) EmptyCollection.emptyCollection) (∀ (i : ι), Membership. …
  -/
  simp only [← not_nonempty_iff_eq_empty, sigma_nonempty, not_exists, not_and]
  /-
    🎉 no goals
  -/


@[mono]
theorem sigma_mono (hs : s₁ ⊆ s₂) (ht : ∀ i, t₁ i ⊆ t₂ i) : s₁.sigma t₁ ⊆ s₂.sigma t₂ :=
  fun ⟨i, _⟩ h =>
  let ⟨hi, ha⟩ := mem_sigma.1 h
  mem_sigma.2 ⟨hs hi, ht i ha⟩


theorem pairwiseDisjoint_map_sigmaMk :
    (s : Set ι).PairwiseDisjoint fun i => (t i).map (Embedding.sigmaMk i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    ⊢ (↑s).PairwiseDisjoint fun i => Finset.map (Function.Embedding.sigmaMk i) (t i)
  -/
  intro i _ j _ hij
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    i : ι
    a✝¹ : Membership.mem (↑s) i
    j : ι
    a✝ : Membership.mem (↑s) j
    hij : Ne i j
    ⊢ Function.onFun Disjoint (fun i => Finset.map (Function.Embedding.sigmaMk i)  …
  -/
  rw [Function.onFun, disjoint_left]
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    i : ι
    a✝¹ : Membership.mem (↑s) i
    j : ι
    a✝ : Membership.mem (↑s) j
    hij : Ne i j
    ⊢ ∀ ⦃a : Sigma fun x => α x⦄, Membership.mem (Finset.map (Function.Embedding.s …
  -/
  simp_rw [mem_map, Function.Embedding.sigmaMk_apply]
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    i : ι
    a✝¹ : Membership.mem (↑s) i
    j : ι
    a✝ : Membership.mem (↑s) j
    hij : Ne i j
    ⊢ ∀ ⦃a : Sigma fun x => α x⦄, (Exists fun a_1 => And (Membership.mem (t i) a_1 …
  -/
  rintro _ ⟨y, _, rfl⟩ ⟨z, _, hz'⟩
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    t : (i : ι) → Finset (α i)
    i : ι
    a✝¹ : Membership.mem (↑s) i
    j : ι
    a✝ : Membership.mem (↑s) j
    hij : Ne i j
    y : α i
    left✝¹ : Membership.mem (t i) y
    z : α j
    left✝ : Membership.mem (t j) z
    hz' : Eq ⟨j, z⟩ ⟨i, y⟩
    ⊢ False
  -/
  exact hij (congr_arg Sigma.fst hz'.symm)
  /-
    🎉 no goals
  -/


@[simp]
theorem disjiUnion_map_sigma_mk :
    s.disjiUnion (fun i => (t i).map (Embedding.sigmaMk i)) pairwiseDisjoint_map_sigmaMk =
      s.sigma t :=
  rfl


theorem sigma_eq_biUnion [DecidableEq (Σi, α i)] (s : Finset ι) (t : ∀ i, Finset (α i)) :
    s.sigma t = s.biUnion fun i => (t i).map <| Embedding.sigmaMk i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : DecidableEq (Sigma fun i => α i)
    s : Finset ι
    t : (i : ι) → Finset (α i)
    ⊢ Eq (s.sigma t) (s.biUnion fun i => Finset.map (Function.Embedding.sigmaMk i) …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : DecidableEq (Sigma fun i => α i)
    s : Finset ι
    t : (i : ι) → Finset (α i)
    x : ι
    y : α x
    ⊢ Iff (Membership.mem (s.sigma t) ⟨x, y⟩) (Membership.mem (s.biUnion fun i =>  …
  -/
  simp [and_left_comm]
  /-
    🎉 no goals
  -/


theorem sup_sigma [SemilatticeSup β] [OrderBot β] :
    (s.sigma t).sup f = s.sup fun i => (t i).sup fun b => f ⟨i, b⟩ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : Type u_3
    s : Finset ι
    t : (i : ι) → Finset (α i)
    f : (Sigma fun i => α i) → β
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    ⊢ Eq ((s.sigma t).sup f) (s.sup fun i => (t i).sup fun b => f ⟨i, b⟩)
  -/
  simp only [le_antisymm_iff, Finset.sup_le_iff, mem_sigma, and_imp, Sigma.forall]
  exact
    ⟨fun i a hi ha => (le_sup hi).trans' <| le_sup (f := fun a => f ⟨i, a⟩) ha, fun i hi a ha =>
      le_sup <| mem_sigma.2 ⟨hi, ha⟩⟩


theorem inf_sigma [SemilatticeInf β] [OrderTop β] :
    (s.sigma t).inf f = s.inf fun i => (t i).inf fun b => f ⟨i, b⟩ :=
  @sup_sigma _ _ βᵒᵈ _ _ _ _ _


theorem _root_.biSup_finsetSigma [CompleteLattice β] (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : Sigma α → β) : ⨆ ij ∈ s.sigma t, f ij = ⨆ (i ∈ s) (j ∈ t i), f ⟨i, j⟩ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : Type u_3
    inst✝ : CompleteLattice β
    s : Finset ι
    t : (i : ι) → Finset (α i)
    f : Sigma α → β
    ⊢ Eq (iSup fun ij => iSup fun h => f ij) (iSup fun i => iSup fun h => iSup fun …
  -/
  simp_rw [← Finset.iSup_coe, Finset.coe_sigma, biSup_sigma]
  /-
    🎉 no goals
  -/


theorem _root_.biSup_finsetSigma' [CompleteLattice β] (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : ∀ i, α i → β) : ⨆ (i ∈ s) (j ∈ t i), f i j = ⨆ ij ∈ s.sigma t, f ij.fst ij.snd :=
  Eq.symm (biSup_finsetSigma _ _ _)


theorem _root_.biInf_finsetSigma [CompleteLattice β] (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : Sigma α → β) : ⨅ ij ∈ s.sigma t, f ij = ⨅ (i ∈ s) (j ∈ t i), f ⟨i, j⟩ :=
  biSup_finsetSigma (β := βᵒᵈ) _ _ _


theorem _root_.biInf_finsetSigma' [CompleteLattice β] (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : ∀ i, α i → β) : ⨅ (i ∈ s) (j ∈ t i), f i j = ⨅ ij ∈ s.sigma t, f ij.fst ij.snd :=
  Eq.symm (biInf_finsetSigma _ _ _)


theorem _root_.Set.biUnion_finsetSigma (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : Sigma α → Set β) : ⋃ ij ∈ s.sigma t, f ij = ⋃ i ∈ s, ⋃ j ∈ t i, f ⟨i, j⟩ :=
  biSup_finsetSigma _ _ _


theorem _root_.Set.biUnion_finsetSigma' (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : ∀ i, α i → Set β) : ⋃ i ∈ s, ⋃ j ∈ t i, f i j = ⋃ ij ∈ s.sigma t, f ij.fst ij.snd :=
  biSup_finsetSigma' _ _ _


theorem _root_.Set.biInter_finsetSigma (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : Sigma α → Set β) : ⋂ ij ∈ s.sigma t, f ij = ⋂ i ∈ s, ⋂ j ∈ t i, f ⟨i, j⟩ :=
  biInf_finsetSigma _ _ _


theorem _root_.Set.biInter_finsetSigma' (s : Finset ι) (t : ∀ i, Finset (α i))
    (f : ∀ i, α i → Set β) : ⋂ i ∈ s, ⋂ j ∈ t i, f i j = ⋂ ij ∈ s.sigma t, f ij.1 ij.2 :=
  biInf_finsetSigma' _ _ _


/-- Lifts maps `α i → β i → Finset (γ i)` to a map `Σ i, α i → Σ i, β i → Finset (Σ i, γ i)`. -/
def sigmaLift (f : ∀ ⦃i⦄, α i → β i → Finset (γ i)) (a : Sigma α) (b : Sigma β) :
    Finset (Sigma γ) :=
  dite (a.1 = b.1) (fun h => (f (h ▸ a.2) b.2).map <| Embedding.sigmaMk _) fun _ => ∅


theorem mem_sigmaLift (f : ∀ ⦃i⦄, α i → β i → Finset (γ i)) (a : Sigma α) (b : Sigma β)
    (x : Sigma γ) :
    x ∈ sigmaLift f a b ↔ ∃ (ha : a.1 = x.1) (hb : b.1 = x.1), x.2 ∈ f (ha ▸ a.2) (hb ▸ b.2) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma α
    b : Sigma β
    x : Sigma γ
    ⊢ Iff (Membership.mem (Finset.sigmaLift f a b) x) (Exists fun ha => Exists fun …
  -/
  obtain ⟨⟨i, a⟩, j, b⟩ := a, b
  /-
    case mk.mk
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    x : Sigma γ
    i : ι
    a : α i
    j : ι
    b : β j
    ⊢ Iff (Membership.mem (Finset.sigmaLift f ⟨i, a⟩ ⟨j, b⟩) x) (Exists fun ha =>  …
  -/
  obtain rfl | h := Decidable.eq_or_ne i j
    /-
      case mk.mk.inl
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      x : Sigma γ
      i : ι
      a : α i
      b : β i
      ⊢ Iff (Membership.mem (Finset.sigmaLift f ⟨i, a⟩ ⟨i, b⟩) x) (Exists fun ha =>  …
    -/
  · constructor
      /-
        case mk.mk.inl.mp
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        x : Sigma γ
        i : ι
        a : α i
        b : β i
        ⊢ Membership.mem (Finset.sigmaLift f ⟨i, a⟩ ⟨i, b⟩) x → Exists fun ha => Exist …
      -/
    · simp_rw [sigmaLift]
      simp only [dite_eq_ite, ite_true, mem_map, Embedding.sigmaMk_apply, forall_exists_index,
        and_imp]
      /-
        case mk.mk.inl.mp
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        x : Sigma γ
        i : ι
        a : α i
        b : β i
        ⊢ ∀ (x_1 : γ i), Membership.mem (f a b) x_1 → Eq ⟨i, x_1⟩ x → Exists fun h =>  …
      -/
      rintro x hx rfl
      /-
        case mk.mk.inl.mp
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        i : ι
        a : α i
        b : β i
        x : γ i
        hx : Membership.mem (f a b) x
        ⊢ Exists fun h => Exists fun h_1 => Membership.mem (f (Eq.rec a ⋯) (Eq.rec b ⋯ …
      -/
      exact ⟨rfl, rfl, hx⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.inl.mpr
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        x : Sigma γ
        i : ι
        a : α i
        b : β i
        ⊢ (Exists fun ha => Exists fun hb => Membership.mem (f (Eq.rec ⟨i, a⟩.snd ha)  …
      -/
    · rintro ⟨⟨⟩, ⟨⟩, hx⟩
      /-
        case mk.mk.inl.mpr.intro.refl.intro.refl
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        x : Sigma γ
        a : α x.1
        b : β x.1
        hx : Membership.mem (f (Eq.rec ⟨x.1, a⟩.snd ⋯) (Eq.rec ⟨x.1, b⟩.snd ⋯)) x.snd
        ⊢ Membership.mem (Finset.sigmaLift f ⟨x.1, a⟩ ⟨x.1, b⟩) x
      -/
      rw [sigmaLift, dif_pos rfl, mem_map]
      /-
        case mk.mk.inl.mpr.intro.refl.intro.refl
        ι : Type u_1
        α : ι → Type u_2
        β : ι → Type u_3
        γ : ι → Type u_4
        inst✝ : DecidableEq ι
        f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
        x : Sigma γ
        a : α x.1
        b : β x.1
        hx : Membership.mem (f (Eq.rec ⟨x.1, a⟩.snd ⋯) (Eq.rec ⟨x.1, b⟩.snd ⋯)) x.snd
        ⊢ Exists fun a_1 => And (Membership.mem (f (Eq.rec ⟨x.1, a⟩.snd ⋯) ⟨x.1, b⟩.sn …
      -/
      exact ⟨_, hx, by simp [Sigma.ext_iff]⟩
      /-
        🎉 no goals
      -/
    /-
      case mk.mk.inr
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      x : Sigma γ
      i : ι
      a : α i
      j : ι
      b : β j
      h : Ne i j
      ⊢ Iff (Membership.mem (Finset.sigmaLift f ⟨i, a⟩ ⟨j, b⟩) x) (Exists fun ha =>  …
    -/
  · rw [sigmaLift, dif_neg h]
    /-
      case mk.mk.inr
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      x : Sigma γ
      i : ι
      a : α i
      j : ι
      b : β j
      h : Ne i j
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection x) (Exists fun ha => Exi …
    -/
    refine iff_of_false (not_mem_empty _) ?_
    /-
      case mk.mk.inr
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      x : Sigma γ
      i : ι
      a : α i
      j : ι
      b : β j
      h : Ne i j
      ⊢ Not (Exists fun ha => Exists fun hb => Membership.mem (f (Eq.rec ⟨i, a⟩.snd  …
    -/
    rintro ⟨⟨⟩, ⟨⟩, _⟩
    /-
      case mk.mk.inr.intro.refl.intro.refl
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      x : Sigma γ
      a : α x.1
      b : β x.1
      h : Ne x.1 x.1
      h✝ : Membership.mem (f (Eq.rec ⟨x.1, a⟩.snd ⋯) (Eq.rec ⟨x.1, b⟩.snd ⋯)) x.snd
      ⊢ False
    -/
    exact h rfl
    /-
      🎉 no goals
    -/


theorem mk_mem_sigmaLift (f : ∀ ⦃i⦄, α i → β i → Finset (γ i)) (i : ι) (a : α i) (b : β i)
    (x : γ i) : (⟨i, x⟩ : Sigma γ) ∈ sigmaLift f ⟨i, a⟩ ⟨i, b⟩ ↔ x ∈ f a b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    i : ι
    a : α i
    b : β i
    x : γ i
    ⊢ Iff (Membership.mem (Finset.sigmaLift f ⟨i, a⟩ ⟨i, b⟩) ⟨i, x⟩) (Membership.m …
  -/
  rw [sigmaLift, dif_pos rfl, mem_map]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    i : ι
    a : α i
    b : β i
    x : γ i
    ⊢ Iff (Exists fun a_1 => And (Membership.mem (f (Eq.rec ⟨i, a⟩.snd ⋯) ⟨i, b⟩.s …
  -/
  refine ⟨?_, fun hx => ⟨_, hx, rfl⟩⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    i : ι
    a : α i
    b : β i
    x : γ i
    ⊢ (Exists fun a_1 => And (Membership.mem (f (Eq.rec ⟨i, a⟩.snd ⋯) ⟨i, b⟩.snd)  …
  -/
  rintro ⟨x, hx, _, rfl⟩
  /-
    case intro.intro.refl
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    i : ι
    a : α i
    b : β i
    x : γ i
    hx : Membership.mem (f (Eq.rec ⟨i, a⟩.snd ⋯) ⟨i, b⟩.snd) x
    ⊢ Membership.mem (f a b) x
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem not_mem_sigmaLift_of_ne_left (f : ∀ ⦃i⦄, α i → β i → Finset (γ i)) (a : Sigma α)
    (b : Sigma β) (x : Sigma γ) (h : a.1 ≠ x.1) : x ∉ sigmaLift f a b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma α
    b : Sigma β
    x : Sigma γ
    h : Ne a.fst x.fst
    ⊢ Not (Membership.mem (Finset.sigmaLift f a b) x)
  -/
  rw [mem_sigmaLift]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma α
    b : Sigma β
    x : Sigma γ
    h : Ne a.fst x.fst
    ⊢ Not (Exists fun ha => Exists fun hb => Membership.mem (f (Eq.rec a.snd ha) ( …
  -/
  exact fun H => h H.fst
  /-
    🎉 no goals
  -/


theorem not_mem_sigmaLift_of_ne_right (f : ∀ ⦃i⦄, α i → β i → Finset (γ i)) {a : Sigma α}
    (b : Sigma β) {x : Sigma γ} (h : b.1 ≠ x.1) : x ∉ sigmaLift f a b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma α
    b : Sigma β
    x : Sigma γ
    h : Ne b.fst x.fst
    ⊢ Not (Membership.mem (Finset.sigmaLift f a b) x)
  -/
  rw [mem_sigmaLift]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma α
    b : Sigma β
    x : Sigma γ
    h : Ne b.fst x.fst
    ⊢ Not (Exists fun ha => Exists fun hb => Membership.mem (f (Eq.rec a.snd ha) ( …
  -/
  exact fun H => h H.snd.fst
  /-
    🎉 no goals
  -/


theorem sigmaLift_nonempty :
    (sigmaLift f a b).Nonempty ↔ ∃ h : a.1 = b.1, (f (h ▸ a.2) b.2).Nonempty := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Iff (Finset.sigmaLift f a b).Nonempty (Exists fun h => (f (Eq.rec a.snd h) b …
  -/
  simp_rw [nonempty_iff_ne_empty, sigmaLift]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Iff (Ne (dite (Eq a.fst b.fst) (fun h => Finset.map (Function.Embedding.sigm …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem sigmaLift_eq_empty : sigmaLift f a b = ∅ ↔ ∀ h : a.1 = b.1, f (h ▸ a.2) b.2 = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Iff (Eq (Finset.sigmaLift f a b) EmptyCollection.emptyCollection) (∀ (h : Eq …
  -/
  simp_rw [sigmaLift]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Iff (Eq (dite (Eq a.fst b.fst) (fun h => Finset.map (Function.Embedding.sigm …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      a : Sigma fun i => α i
      b : Sigma fun i => β i
      h : Eq a.fst b.fst
      ⊢ Iff (Eq (Finset.map (Function.Embedding.sigmaMk b.fst) (f (Eq.rec a.snd h) b …
    -/
  · simp [h, forall_prop_of_true h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      γ : ι → Type u_4
      inst✝ : DecidableEq ι
      f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
      a : Sigma fun i => α i
      b : Sigma fun i => β i
      h : Not (Eq a.fst b.fst)
      ⊢ Iff (Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection) (∀  …
    -/
  · simp [h, forall_prop_of_false h]
    /-
      🎉 no goals
    -/


theorem sigmaLift_mono (h : ∀ ⦃i⦄ ⦃a : α i⦄ ⦃b : β i⦄, f a b ⊆ g a b) (a : Σi, α i) (b : Σi, β i) :
    sigmaLift f a b ⊆ sigmaLift g a b := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f g : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    h : ∀ ⦃i : ι⦄ ⦃a : α i⦄ ⦃b : β i⦄, HasSubset.Subset (f a b) (g a b)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ HasSubset.Subset (Finset.sigmaLift f a b) (Finset.sigmaLift g a b)
  -/
  rintro x hx
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f g : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    h : ∀ ⦃i : ι⦄ ⦃a : α i⦄ ⦃b : β i⦄, HasSubset.Subset (f a b) (g a b)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    x : Sigma γ
    hx : Membership.mem (Finset.sigmaLift f a b) x
    ⊢ Membership.mem (Finset.sigmaLift g a b) x
  -/
  rw [mem_sigmaLift] at hx ⊢
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f g : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    h : ∀ ⦃i : ι⦄ ⦃a : α i⦄ ⦃b : β i⦄, HasSubset.Subset (f a b) (g a b)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    x : Sigma γ
    hx : Exists fun ha => Exists fun hb => Membership.mem (f (Eq.rec a.snd ha) (Eq …
    ⊢ Exists fun ha => Exists fun hb => Membership.mem (g (Eq.rec a.snd ha) (Eq.re …
  -/
  obtain ⟨ha, hb, hx⟩ := hx
  /-
    case intro.intro
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f g : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    h : ∀ ⦃i : ι⦄ ⦃a : α i⦄ ⦃b : β i⦄, HasSubset.Subset (f a b) (g a b)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    x : Sigma γ
    ha : Eq a.fst x.fst
    hb : Eq b.fst x.fst
    hx : Membership.mem (f (Eq.rec a.snd ha) (Eq.rec b.snd hb)) x.snd
    ⊢ Exists fun ha => Exists fun hb => Membership.mem (g (Eq.rec a.snd ha) (Eq.re …
  -/
  exact ⟨ha, hb, h hx⟩
  /-
    🎉 no goals
  -/


theorem card_sigmaLift :
    (sigmaLift f a b).card = dite (a.1 = b.1) (fun h => (f (h ▸ a.2) b.2).card) fun _ => 0 := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Eq (Finset.sigmaLift f a b).card (dite (Eq a.fst b.fst) (fun h => (f (Eq.rec …
  -/
  simp_rw [sigmaLift]
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    γ : ι → Type u_4
    inst✝ : DecidableEq ι
    f : ⦃i : ι⦄ → α i → β i → Finset (γ i)
    a : Sigma fun i => α i
    b : Sigma fun i => β i
    ⊢ Eq (dite (Eq a.fst b.fst) (fun h => Finset.map (Function.Embedding.sigmaMk b …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


