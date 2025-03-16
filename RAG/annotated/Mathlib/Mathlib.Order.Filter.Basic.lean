instance inhabitedMem : Inhabited { s : Set α // s ∈ f } :=
  ⟨⟨univ, f.univ_sets⟩⟩


theorem filter_eq_iff : f = g ↔ f.sets = g.sets :=
  ⟨congr_arg _, filter_eq⟩


@[simp] theorem sets_subset_sets : f.sets ⊆ g.sets ↔ g ≤ f := .rfl

@[simp] theorem sets_ssubset_sets : f.sets ⊂ g.sets ↔ g < f := .rfl


/-- An extensionality lemma that is useful for filters with good lemmas about `sᶜ ∈ f` (e.g.,
`Filter.comap`, `Filter.coprod`, `Filter.Coprod`, `Filter.cofinite`). -/
protected theorem coext (h : ∀ s, sᶜ ∈ f ↔ sᶜ ∈ g) : f = g :=
  Filter.ext <| compl_surjective.forall.2 h


instance : Trans (· ⊇ ·) ((· ∈ ·) : Set α → Filter α → Prop) (· ∈ ·) where
  trans h₁ h₂ := mem_of_superset h₂ h₁


instance : Trans Membership.mem (· ⊆ ·) (Membership.mem : Filter α → Set α → Prop) where
  trans h₁ h₂ := mem_of_superset h₁ h₂


@[simp]
theorem inter_mem_iff {s t : Set α} : s ∩ t ∈ f ↔ s ∈ f ∧ t ∈ f :=
  ⟨fun h => ⟨mem_of_superset h inter_subset_left, mem_of_superset h inter_subset_right⟩,
    and_imp.2 inter_mem⟩


theorem diff_mem {s t : Set α} (hs : s ∈ f) (ht : tᶜ ∈ f) : s \ t ∈ f :=
  inter_mem hs ht


theorem congr_sets (h : { x | x ∈ s ↔ x ∈ t } ∈ f) : s ∈ f ↔ t ∈ f :=
  ⟨fun hs => mp_mem hs (mem_of_superset h fun _ => Iff.mp), fun hs =>
    mp_mem hs (mem_of_superset h fun _ => Iff.mpr)⟩


lemma copy_eq {S} (hmem : ∀ s, s ∈ S ↔ s ∈ f) : f.copy S hmem = f := Filter.ext hmem


/-- Weaker version of `Filter.biInter_mem` that assumes `Subsingleton β` rather than `Finite β`. -/
theorem biInter_mem' {β : Type v} {s : β → Set α} {is : Set β} (hf : is.Subsingleton) :
    (⋂ i ∈ is, s i) ∈ f ↔ ∀ i ∈ is, s i ∈ f := by
  /-
    α : Type u
    f : Filter α
    β : Type v
    s : β → Set α
    is : Set β
    hf : is.Subsingleton
    ⊢ Iff (Membership.mem f (Set.iInter fun i => Set.iInter fun h => s i)) (∀ (i : …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  apply Subsingleton.induction_on hf <;> simp
                                         /-
                                           🎉 no goals
                                         -/


/-- Weaker version of `Filter.iInter_mem` that assumes `Subsingleton β` rather than `Finite β`. -/
theorem iInter_mem' {β : Sort v} {s : β → Set α} [Subsingleton β] :
    (⋂ i, s i) ∈ f ↔ ∀ i, s i ∈ f := by
  /-
    α : Type u
    f : Filter α
    β : Sort v
    s : β → Set α
    inst✝ : Subsingleton β
    ⊢ Iff (Membership.mem f (Set.iInter fun i => s i)) (∀ (i : β), Membership.mem  …
  -/
  rw [← sInter_range, sInter_eq_biInter, biInter_mem' (subsingleton_range s), forall_mem_range]
  /-
    🎉 no goals
  -/


theorem exists_mem_subset_iff : (∃ t ∈ f, t ⊆ s) ↔ s ∈ f :=
  ⟨fun ⟨_, ht, ts⟩ => mem_of_superset ht ts, fun hs => ⟨s, hs, Subset.rfl⟩⟩


theorem monotone_mem {f : Filter α} : Monotone fun s => s ∈ f := fun _ _ hst h =>
  mem_of_superset h hst


theorem exists_mem_and_iff {P : Set α → Prop} {Q : Set α → Prop} (hP : Antitone P)
    (hQ : Antitone Q) : ((∃ u ∈ f, P u) ∧ ∃ u ∈ f, Q u) ↔ ∃ u ∈ f, P u ∧ Q u := by
  /-
    α : Type u
    f : Filter α
    P Q : Set α → Prop
    hP : Antitone P
    hQ : Antitone Q
    ⊢ Iff (And (Exists fun u => And (Membership.mem f u) (P u)) (Exists fun u => A …
  -/
  constructor
    /-
      case mp
      α : Type u
      f : Filter α
      P Q : Set α → Prop
      hP : Antitone P
      hQ : Antitone Q
      ⊢ And (Exists fun u => And (Membership.mem f u) (P u)) (Exists fun u => And (M …
    -/
  · rintro ⟨⟨u, huf, hPu⟩, v, hvf, hQv⟩
    exact
      ⟨u ∩ v, inter_mem huf hvf, hP inter_subset_left hPu, hQ inter_subset_right hQv⟩
    /-
      case mpr
      α : Type u
      f : Filter α
      P Q : Set α → Prop
      hP : Antitone P
      hQ : Antitone Q
      ⊢ (Exists fun u => And (Membership.mem f u) (And (P u) (Q u))) → And (Exists f …
    -/
  · rintro ⟨u, huf, hPu, hQu⟩
    /-
      case mpr.intro.intro.intro
      α : Type u
      f : Filter α
      P Q : Set α → Prop
      hP : Antitone P
      hQ : Antitone Q
      u : Set α
      huf : Membership.mem f u
      hPu : P u
      hQu : Q u
      ⊢ And (Exists fun u => And (Membership.mem f u) (P u)) (Exists fun u => And (M …
    -/
    exact ⟨⟨u, huf, hPu⟩, u, huf, hQu⟩
    /-
      🎉 no goals
    -/


theorem forall_in_swap {β : Type*} {p : Set α → β → Prop} :
    (∀ a ∈ f, ∀ (b), p a b) ↔ ∀ (b), ∀ a ∈ f, p a b :=
  Set.forall_in_swap


theorem mem_principal_self (s : Set α) : s ∈ 𝓟 s := Subset.rfl


                                                         /-
                                                           α : Type u
                                                           f g : Filter α
                                                           ⊢ Iff (Not (LE.le f g)) (Exists fun s => And (Membership.mem g s) (Not (Member …
                                                         -/
protected theorem not_le : ¬f ≤ g ↔ ∃ s ∈ g, s ∉ f := by simp_rw [le_def, not_forall, exists_prop]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- `GenerateSets g s`: `s` is in the filter closure of `g`. -/
inductive GenerateSets (g : Set (Set α)) : Set α → Prop
  | basic {s : Set α} : s ∈ g → GenerateSets g s
  | univ : GenerateSets g univ
  | superset {s t : Set α} : GenerateSets g s → s ⊆ t → GenerateSets g t
  | inter {s t : Set α} : GenerateSets g s → GenerateSets g t → GenerateSets g (s ∩ t)


/-- `generate g` is the largest filter containing the sets `g`. -/
def generate (g : Set (Set α)) : Filter α where
  sets := {s | GenerateSets g s}
  univ_sets := GenerateSets.univ
  sets_of_superset := GenerateSets.superset
  inter_sets := GenerateSets.inter


lemma mem_generate_of_mem {s : Set <| Set α} {U : Set α} (h : U ∈ s) :
    U ∈ generate s := GenerateSets.basic h


theorem le_generate_iff {s : Set (Set α)} {f : Filter α} : f ≤ generate s ↔ s ⊆ f.sets :=
  Iff.intro (fun h _ hu => h <| GenerateSets.basic <| hu) fun h _ hu =>
    hu.recOn (fun h' => h h') univ_mem (fun _ hxy hx => mem_of_superset hx hxy) fun _ _ hx hy =>
      inter_mem hx hy


@[simp] lemma generate_singleton (s : Set α) : generate {s} = 𝓟 s :=
  le_antisymm (fun _t ht ↦ mem_of_superset (mem_generate_of_mem <| mem_singleton _) ht) <|
    le_generate_iff.2 <| singleton_subset_iff.2 Subset.rfl


/-- `mkOfClosure s hs` constructs a filter on `α` whose elements set is exactly
`s : Set (Set α)`, provided one gives the assumption `hs : (generate s).sets = s`. -/
protected def mkOfClosure (s : Set (Set α)) (hs : (generate s).sets = s) : Filter α where
  sets := s
  univ_sets := hs ▸ univ_mem
  sets_of_superset := hs ▸ mem_of_superset
  inter_sets := hs ▸ inter_mem


theorem mkOfClosure_sets {s : Set (Set α)} {hs : (generate s).sets = s} :
    Filter.mkOfClosure s hs = generate s :=
  Filter.ext fun u =>
    show u ∈ (Filter.mkOfClosure s hs).sets ↔ u ∈ (generate s).sets from hs.symm ▸ Iff.rfl


/-- Galois insertion from sets of sets into filters. -/
def giGenerate (α : Type*) :
    @GaloisInsertion (Set (Set α)) (Filter α)ᵒᵈ _ _ Filter.generate Filter.sets where
  gc _ _ := le_generate_iff
  le_l_u _ _ h := GenerateSets.basic h
  choice s hs := Filter.mkOfClosure s (le_antisymm hs <| le_generate_iff.1 <| le_rfl)
  choice_eq _ _ := mkOfClosure_sets


theorem mem_inf_iff {f g : Filter α} {s : Set α} : s ∈ f ⊓ g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, s = t₁ ∩ t₂ :=
  Iff.rfl


theorem mem_inf_of_left {f g : Filter α} {s : Set α} (h : s ∈ f) : s ∈ f ⊓ g :=
  ⟨s, h, univ, univ_mem, (inter_univ s).symm⟩


theorem mem_inf_of_right {f g : Filter α} {s : Set α} (h : s ∈ g) : s ∈ f ⊓ g :=
  ⟨univ, univ_mem, s, h, (univ_inter s).symm⟩


theorem inter_mem_inf {α : Type u} {f g : Filter α} {s t : Set α} (hs : s ∈ f) (ht : t ∈ g) :
    s ∩ t ∈ f ⊓ g :=
  ⟨s, hs, t, ht, rfl⟩


theorem mem_inf_of_inter {f g : Filter α} {s t u : Set α} (hs : s ∈ f) (ht : t ∈ g)
    (h : s ∩ t ⊆ u) : u ∈ f ⊓ g :=
  mem_of_superset (inter_mem_inf hs ht) h


theorem mem_inf_iff_superset {f g : Filter α} {s : Set α} :
    s ∈ f ⊓ g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ ∩ t₂ ⊆ s :=
  ⟨fun ⟨t₁, h₁, t₂, h₂, Eq⟩ => ⟨t₁, h₁, t₂, h₂, Eq ▸ Subset.rfl⟩, fun ⟨_, h₁, _, h₂, sub⟩ =>
    mem_inf_of_inter h₁ h₂ sub⟩


/-- Complete lattice structure on `Filter α`. -/
instance instCompleteLatticeFilter : CompleteLattice (Filter α) where
  inf a b := min a b
  sup a b := max a b
  le_sup_left _ _ _ h := h.1
  le_sup_right _ _ _ h := h.2
  sup_le _ _ _ h₁ h₂ _ h := ⟨h₁ h, h₂ h⟩
  inf_le_left _ _ _ := mem_inf_of_left
  inf_le_right _ _ _ := mem_inf_of_right
  le_inf := fun _ _ _ h₁ h₂ _s ⟨_a, ha, _b, hb, hs⟩ => hs.symm ▸ inter_mem (h₁ ha) (h₂ hb)
  le_sSup _ _ h₁ _ h₂ := h₂ h₁
  sSup_le _ _ h₁ _ h₂ _ h₃ := h₁ _ h₃ h₂
                            /-
                              α : Type u
                              β : Type v
                              γ : Type w
                              δ : Type u_1
                              ι : Sort x
                              f g : Filter α
                              s t : Set α
                              x✝² : Set (Filter α)
                              x✝¹ : Filter α
                              h₁ : Membership.mem x✝² x✝¹
                              x✝ : Set α
                              h₂ : Membership.mem x✝¹ x✝
                              ⊢ Membership.mem (InfSet.sInf x✝²) x✝
                            -/
  sInf_le _ _ h₁ _ h₂ := by rw [← Filter.sSup_lowerBounds]; exact fun _ h₃ ↦ h₃ h₁ h₂
                                                            /-
                                                              🎉 no goals
                                                            -/
                            /-
                              α : Type u
                              β : Type v
                              γ : Type w
                              δ : Type u_1
                              ι : Sort x
                              f g : Filter α
                              s t : Set α
                              x✝² : Set (Filter α)
                              x✝¹ : Filter α
                              h₁ : ∀ (b : Filter α), Membership.mem x✝² b → LE.le x✝¹ b
                              x✝ : Set α
                              h₂ : Membership.mem (InfSet.sInf x✝²) x✝
                              ⊢ Membership.mem x✝¹ x✝
                            -/
  le_sInf _ _ h₁ _ h₂ := by rw [← Filter.sSup_lowerBounds] at h₂; exact h₂ h₁
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  le_top _ _ := univ_mem'
  bot_le _ _ _ := trivial


instance : Inhabited (Filter α) := ⟨⊥⟩


theorem NeBot.ne {f : Filter α} (hf : NeBot f) : f ≠ ⊥ := hf.ne'


@[simp] theorem not_neBot {f : Filter α} : ¬f.NeBot ↔ f = ⊥ := neBot_iff.not_left


theorem NeBot.mono {f g : Filter α} (hf : NeBot f) (hg : f ≤ g) : NeBot g :=
  ⟨ne_bot_of_le_ne_bot hf.1 hg⟩


theorem neBot_of_le {f g : Filter α} [hf : NeBot f] (hg : f ≤ g) : NeBot g :=
  hf.mono hg


@[simp] theorem sup_neBot {f g : Filter α} : NeBot (f ⊔ g) ↔ NeBot f ∨ NeBot g := by
  /-
    α : Type u
    f g : Filter α
    ⊢ Iff (Max.max f g).NeBot (Or f.NeBot g.NeBot)
  -/
  simp only [neBot_iff, not_and_or, Ne, sup_eq_bot_iff]
  /-
    🎉 no goals
  -/


                                                              /-
                                                                α : Type u
                                                                f : Filter α
                                                                ⊢ Iff (Not (Disjoint f f)) f.NeBot
                                                              -/
theorem not_disjoint_self_iff : ¬Disjoint f f ↔ f.NeBot := by rw [disjoint_self, neBot_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem bot_sets_eq : (⊥ : Filter α).sets = univ := rfl


/-- Either `f = ⊥` or `Filter.NeBot f`. This is a version of `eq_or_ne` that uses `Filter.NeBot`
as the second alternative, to be used as an instance. -/
theorem eq_or_neBot (f : Filter α) : f = ⊥ ∨ NeBot f := (eq_or_ne f ⊥).imp_right NeBot.mk


theorem sup_sets_eq {f g : Filter α} : (f ⊔ g).sets = f.sets ∩ g.sets :=
  (giGenerate α).gc.u_inf


theorem sSup_sets_eq {s : Set (Filter α)} : (sSup s).sets = ⋂ f ∈ s, (f : Filter α).sets :=
  (giGenerate α).gc.u_sInf


theorem iSup_sets_eq {f : ι → Filter α} : (iSup f).sets = ⋂ i, (f i).sets :=
  (giGenerate α).gc.u_iInf


theorem generate_empty : Filter.generate ∅ = (⊤ : Filter α) :=
  (giGenerate α).gc.l_bot


theorem generate_univ : Filter.generate univ = (⊥ : Filter α) :=
  bot_unique fun _ _ => GenerateSets.basic (mem_univ _)


theorem generate_union {s t : Set (Set α)} :
    Filter.generate (s ∪ t) = Filter.generate s ⊓ Filter.generate t :=
  (giGenerate α).gc.l_sup


theorem generate_iUnion {s : ι → Set (Set α)} :
    Filter.generate (⋃ i, s i) = ⨅ i, Filter.generate (s i) :=
  (giGenerate α).gc.l_iSup


@[simp]
theorem mem_sup {f g : Filter α} {s : Set α} : s ∈ f ⊔ g ↔ s ∈ f ∧ s ∈ g :=
  Iff.rfl


theorem union_mem_sup {f g : Filter α} {s t : Set α} (hs : s ∈ f) (ht : t ∈ g) : s ∪ t ∈ f ⊔ g :=
  ⟨mem_of_superset hs subset_union_left, mem_of_superset ht subset_union_right⟩


@[simp]
theorem mem_iSup {x : Set α} {f : ι → Filter α} : x ∈ iSup f ↔ ∀ i, x ∈ f i := by
  /-
    α : Type u
    ι : Sort x
    x : Set α
    f : ι → Filter α
    ⊢ Iff (Membership.mem (iSup f) x) (∀ (i : ι), Membership.mem (f i) x)
  -/
  simp only [← Filter.mem_sets, iSup_sets_eq, mem_iInter]
  /-
    🎉 no goals
  -/


@[simp]
theorem iSup_neBot {f : ι → Filter α} : (⨆ i, f i).NeBot ↔ ∃ i, (f i).NeBot := by
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    ⊢ Iff (iSup fun i => f i).NeBot (Exists fun i => (f i).NeBot)
  -/
  simp [neBot_iff]
  /-
    🎉 no goals
  -/


theorem iInf_eq_generate (s : ι → Filter α) : iInf s = generate (⋃ i, (s i).sets) :=
                                 /-
                                   α : Type u
                                   ι : Sort x
                                   s : ι → Filter α
                                   x✝ : Filter α
                                   ⊢ Iff (LE.le x✝ (iInf s)) (LE.le x✝ (Filter.generate (Set.iUnion fun i => (s i …
                                 -/
  eq_of_forall_le_iff fun _ ↦ by simp [le_generate_iff]
                                 /-
                                   🎉 no goals
                                 -/


theorem mem_iInf_of_mem {f : ι → Filter α} (i : ι) {s} (hs : s ∈ f i) : s ∈ ⨅ i, f i :=
  iInf_le f i hs


@[simp]
theorem le_principal_iff {s : Set α} {f : Filter α} : f ≤ 𝓟 s ↔ s ∈ f :=
  ⟨fun h => h Subset.rfl, fun hs _ ht => mem_of_superset hs ht⟩


theorem Iic_principal (s : Set α) : Iic (𝓟 s) = { l | s ∈ l } :=
  Set.ext fun _ => le_principal_iff


theorem principal_mono {s t : Set α} : 𝓟 s ≤ 𝓟 t ↔ s ⊆ t := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (LE.le (Filter.principal s) (Filter.principal t)) (HasSubset.Subset s t)
  -/
  simp only [le_principal_iff, mem_principal]
  /-
    🎉 no goals
  -/


@[gcongr] alias ⟨_, _root_.GCongr.filter_principal_mono⟩ := principal_mono


@[mono]
theorem monotone_principal : Monotone (𝓟 : Set α → Filter α) := fun _ _ => principal_mono.2


@[simp] theorem principal_eq_iff_eq {s t : Set α} : 𝓟 s = 𝓟 t ↔ s = t := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Eq (Filter.principal s) (Filter.principal t)) (Eq s t)
  -/
  simp only [le_antisymm_iff, le_principal_iff, mem_principal]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp] theorem join_principal_eq_sSup {s : Set (Filter α)} : join (𝓟 s) = sSup s := rfl


@[simp] theorem principal_univ : 𝓟 (univ : Set α) = ⊤ :=
                   /-
                     α : Type u
                     ⊢ LE.le Top.top (Filter.principal Set.univ)
                   -/
  top_unique <| by simp only [le_principal_iff, mem_top, eq_self_iff_true]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem principal_empty : 𝓟 (∅ : Set α) = ⊥ :=
  bot_unique fun _ _ => empty_subset _


theorem generate_eq_biInf (S : Set (Set α)) : generate S = ⨅ s ∈ S, 𝓟 s :=
                                  /-
                                    α : Type u
                                    S : Set (Set α)
                                    f : Filter α
                                    ⊢ Iff (LE.le f (Filter.generate S)) (LE.le f (iInf fun s => iInf fun h => Filt …
                                  -/
  eq_of_forall_le_iff fun f => by simp [le_generate_iff, le_principal_iff, subset_def]
                                  /-
                                    🎉 no goals
                                  -/


theorem empty_mem_iff_bot {f : Filter α} : ∅ ∈ f ↔ f = ⊥ :=
  ⟨fun h => bot_unique fun s _ => mem_of_superset h (empty_subset s), fun h => h.symm ▸ mem_bot⟩


theorem nonempty_of_mem {f : Filter α} [hf : NeBot f] {s : Set α} (hs : s ∈ f) : s.Nonempty :=
  s.eq_empty_or_nonempty.elim (fun h => absurd hs (h.symm ▸ mt empty_mem_iff_bot.mp hf.1)) id


theorem NeBot.nonempty_of_mem {f : Filter α} (hf : NeBot f) {s : Set α} (hs : s ∈ f) : s.Nonempty :=
  @Filter.nonempty_of_mem α f hf s hs


@[simp]
theorem empty_not_mem (f : Filter α) [NeBot f] : ¬∅ ∈ f := fun h => (nonempty_of_mem h).ne_empty rfl


theorem nonempty_of_neBot (f : Filter α) [NeBot f] : Nonempty α :=
  nonempty_of_exists <| nonempty_of_mem (univ_mem : univ ∈ f)


theorem compl_not_mem {f : Filter α} {s : Set α} [NeBot f] (h : s ∈ f) : sᶜ ∉ f := fun hsc =>
  (nonempty_of_mem (inter_mem h hsc)).ne_empty <| inter_compl_self s


theorem filter_eq_bot_of_isEmpty [IsEmpty α] (f : Filter α) : f = ⊥ :=
  empty_mem_iff_bot.mp <| univ_mem' isEmptyElim


protected lemma disjoint_iff {f g : Filter α} : Disjoint f g ↔ ∃ s ∈ f, ∃ t ∈ g, Disjoint s t := by
  simp only [disjoint_iff, ← empty_mem_iff_bot, mem_inf_iff, inf_eq_inter, bot_eq_empty,
    @eq_comm _ ∅]


theorem disjoint_of_disjoint_of_mem {f g : Filter α} {s t : Set α} (h : Disjoint s t) (hs : s ∈ f)
    (ht : t ∈ g) : Disjoint f g :=
  Filter.disjoint_iff.mpr ⟨s, hs, t, ht, h⟩


theorem NeBot.not_disjoint (hf : f.NeBot) (hs : s ∈ f) (ht : t ∈ f) : ¬Disjoint s t := fun h =>
  not_disjoint_self_iff.2 hf <| Filter.disjoint_iff.2 ⟨s, hs, t, ht, h⟩


theorem inf_eq_bot_iff {f g : Filter α} : f ⊓ g = ⊥ ↔ ∃ U ∈ f, ∃ V ∈ g, U ∩ V = ∅ := by
  /-
    α : Type u
    f g : Filter α
    ⊢ Iff (Eq (Min.min f g) Bot.bot) (Exists fun U => And (Membership.mem f U) (Ex …
  -/
  simp only [← disjoint_iff, Filter.disjoint_iff, Set.disjoint_iff_inter_eq_empty]
  /-
    🎉 no goals
  -/


/-- There is exactly one filter on an empty type. -/
instance unique [IsEmpty α] : Unique (Filter α) where
  default := ⊥
  uniq := filter_eq_bot_of_isEmpty


theorem NeBot.nonempty (f : Filter α) [hf : f.NeBot] : Nonempty α :=
  not_isEmpty_iff.mp fun _ ↦ hf.ne (Subsingleton.elim _ _)


/-- There are only two filters on a `Subsingleton`: `⊥` and `⊤`. If the type is empty, then they are
equal. -/
theorem eq_top_of_neBot [Subsingleton α] (l : Filter α) [NeBot l] : l = ⊤ := by
  /-
    α : Type u
    inst✝¹ : Subsingleton α
    l : Filter α
    inst✝ : l.NeBot
    ⊢ Eq l Top.top
  -/
  refine top_unique fun s hs => ?_
  /-
    α : Type u
    inst✝¹ : Subsingleton α
    l : Filter α
    inst✝ : l.NeBot
    s : Set α
    hs : Membership.mem l s
    ⊢ Membership.mem Top.top s
  -/
  obtain rfl : s = univ := Subsingleton.eq_univ_of_nonempty (nonempty_of_mem hs)
  /-
    α : Type u
    inst✝¹ : Subsingleton α
    l : Filter α
    inst✝ : l.NeBot
    hs : Membership.mem l Set.univ
    ⊢ Membership.mem Top.top Set.univ
  -/
  exact univ_mem
  /-
    🎉 no goals
  -/


theorem forall_mem_nonempty_iff_neBot {f : Filter α} :
    (∀ s : Set α, s ∈ f → s.Nonempty) ↔ NeBot f :=
  ⟨fun h => ⟨fun hf => not_nonempty_empty (h ∅ <| hf.symm ▸ mem_bot)⟩, @nonempty_of_mem _ _⟩


instance instNontrivialFilter [Nonempty α] : Nontrivial (Filter α) :=
  ⟨⟨⊤, ⊥, NeBot.ne <| forall_mem_nonempty_iff_neBot.1
                   /-
                     α : Type u
                     β : Type v
                     γ : Type w
                     δ : Type u_1
                     ι : Sort x
                     f g : Filter α
                     s✝ t : Set α
                     inst✝ : Nonempty α
                     s : Set α
                     hs : Membership.mem Top.top s
                     ⊢ s.Nonempty
                   -/
    fun s hs => by rwa [mem_top.1 hs, ← nonempty_iff_univ_nonempty]⟩⟩
                   /-
                     🎉 no goals
                   -/


theorem nontrivial_iff_nonempty : Nontrivial (Filter α) ↔ Nonempty α :=
  ⟨fun _ =>
    by_contra fun h' =>
      haveI := not_nonempty_iff.1 h'
      not_subsingleton (Filter α) inferInstance,
    @Filter.instNontrivialFilter α⟩


theorem eq_sInf_of_mem_iff_exists_mem {S : Set (Filter α)} {l : Filter α}
    (h : ∀ {s}, s ∈ l ↔ ∃ f ∈ S, s ∈ f) : l = sInf S :=
  le_antisymm (le_sInf fun f hf _ hs => h.2 ⟨f, hf, hs⟩)
    fun _ hs => let ⟨_, hf, hs⟩ := h.1 hs; (sInf_le hf) hs


theorem eq_iInf_of_mem_iff_exists_mem {f : ι → Filter α} {l : Filter α}
    (h : ∀ {s}, s ∈ l ↔ ∃ i, s ∈ f i) : l = iInf f :=
  eq_sInf_of_mem_iff_exists_mem <| h.trans (exists_range_iff (p := (_ ∈ ·))).symm


theorem eq_biInf_of_mem_iff_exists_mem {f : ι → Filter α} {p : ι → Prop} {l : Filter α}
    (h : ∀ {s}, s ∈ l ↔ ∃ i, p i ∧ s ∈ f i) : l = ⨅ (i) (_ : p i), f i := by
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    p : ι → Prop
    l : Filter α
    h : ∀ {s : Set α}, Iff (Membership.mem l s) (Exists fun i => And (p i) (Member …
    ⊢ Eq l (iInf fun i => iInf fun x => f i)
  -/
  rw [iInf_subtype']
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    p : ι → Prop
    l : Filter α
    h : ∀ {s : Set α}, Iff (Membership.mem l s) (Exists fun i => And (p i) (Member …
    ⊢ Eq l (iInf fun x => f ↑x)
  -/
  exact eq_iInf_of_mem_iff_exists_mem fun {_} => by simp only [Subtype.exists, h, exists_prop]
  /-
    🎉 no goals
  -/


theorem iInf_sets_eq {f : ι → Filter α} (h : Directed (· ≥ ·) f) [ne : Nonempty ι] :
    (iInf f).sets = ⋃ i, (f i).sets :=
  let ⟨i⟩ := ne
  let u :=
    { sets := ⋃ i, (f i).sets
      univ_sets := mem_iUnion.2 ⟨i, univ_mem⟩
      sets_of_superset := by
        /-
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          ⊢ ∀ {x y : Set α}, Membership.mem (Set.iUnion fun i => (f i).sets) x → HasSubs …
        -/
        simp only [mem_iUnion, exists_imp]
        /-
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          ⊢ ∀ {x y : Set α} (x_1 : ι), Membership.mem (f x_1).sets x → HasSubset.Subset  …
        -/
        exact fun i hx hxy => ⟨i, mem_of_superset hx hxy⟩
        /-
          🎉 no goals
        -/
      inter_sets := by
        /-
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          ⊢ ∀ {x y : Set α}, Membership.mem (Set.iUnion fun i => (f i).sets) x → Members …
        -/
        simp only [mem_iUnion, exists_imp]
        /-
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          ⊢ ∀ {x y : Set α} (x_1 : ι), Membership.mem (f x_1).sets x → ∀ (x_2 : ι), Memb …
        -/
        intro x y a hx b hy
        /-
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          x y : Set α
          a : ι
          hx : Membership.mem (f a).sets x
          b : ι
          hy : Membership.mem (f b).sets y
          ⊢ Exists fun i => Membership.mem (f i).sets (Inter.inter x y)
        -/
        rcases h a b with ⟨c, ha, hb⟩
        /-
          case intro.intro
          α : Type u
          ι : Sort x
          f : ι → Filter α
          h : Directed (fun x1 x2 => GE.ge x1 x2) f
          ne : Nonempty ι
          i : ι
          x y : Set α
          a : ι
          hx : Membership.mem (f a).sets x
          b : ι
          hy : Membership.mem (f b).sets y
          c : ι
          ha : GE.ge (f a) (f c)
          hb : GE.ge (f b) (f c)
          ⊢ Exists fun i => Membership.mem (f i).sets (Inter.inter x y)
        -/
        exact ⟨c, inter_mem (ha hx) (hb hy)⟩ }
        /-
          🎉 no goals
        -/
  have : u = iInf f := eq_iInf_of_mem_iff_exists_mem mem_iUnion
  -- Porting note: it was just `congr_arg filter.sets this.symm`
                                                /-
                                                  α : Type u
                                                  ι : Sort x
                                                  f : ι → Filter α
                                                  h : Directed (fun x1 x2 => GE.ge x1 x2) f
                                                  ne : Nonempty ι
                                                  i : ι
                                                  u : Filter α := { sets := Set.iUnion fun i => (f i).sets, univ_sets := ⋯, sets …
                                                  this : Eq u (iInf f)
                                                  ⊢ Eq u.sets (Set.iUnion fun i => (f i).sets)
                                                -/
  (congr_arg Filter.sets this.symm).trans <| by simp only [u]
                                                /-
                                                  🎉 no goals
                                                -/


theorem mem_iInf_of_directed {f : ι → Filter α} (h : Directed (· ≥ ·) f) [Nonempty ι] (s) :
    s ∈ iInf f ↔ ∃ i, s ∈ f i := by
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    h : Directed (fun x1 x2 => GE.ge x1 x2) f
    inst✝ : Nonempty ι
    s : Set α
    ⊢ Iff (Membership.mem (iInf f) s) (Exists fun i => Membership.mem (f i) s)
  -/
  simp only [← Filter.mem_sets, iInf_sets_eq h, mem_iUnion]
  /-
    🎉 no goals
  -/


theorem mem_biInf_of_directed {f : β → Filter α} {s : Set β} (h : DirectedOn (f ⁻¹'o (· ≥ ·)) s)
    (ne : s.Nonempty) {t : Set α} : (t ∈ ⨅ i ∈ s, f i) ↔ ∃ i ∈ s, t ∈ f i := by
  /-
    α : Type u
    β : Type v
    f : β → Filter α
    s : Set β
    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) s
    ne : s.Nonempty
    t : Set α
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => f i) t) (Exists fun i => An …
  -/
  haveI := ne.to_subtype
  /-
    α : Type u
    β : Type v
    f : β → Filter α
    s : Set β
    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) s
    ne : s.Nonempty
    t : Set α
    this : Nonempty ↑s
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => f i) t) (Exists fun i => An …
  -/
  simp_rw [iInf_subtype', mem_iInf_of_directed h.directed_val, Subtype.exists, exists_prop]
  /-
    🎉 no goals
  -/


theorem biInf_sets_eq {f : β → Filter α} {s : Set β} (h : DirectedOn (f ⁻¹'o (· ≥ ·)) s)
    (ne : s.Nonempty) : (⨅ i ∈ s, f i).sets = ⋃ i ∈ s, (f i).sets :=
                  /-
                    α : Type u
                    β : Type v
                    f : β → Filter α
                    s : Set β
                    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) s
                    ne : s.Nonempty
                    t : Set α
                    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => f i).sets t) (Membership.me …
                  -/
  ext fun t => by simp [mem_biInf_of_directed h ne]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem sup_join {f₁ f₂ : Filter (Filter α)} : join f₁ ⊔ join f₂ = join (f₁ ⊔ f₂) :=
                         /-
                           α : Type u
                           f₁ f₂ : Filter (Filter α)
                           x : Set α
                           ⊢ Iff (Membership.mem (Max.max f₁.join f₂.join) x) (Membership.mem (Max.max f₁ …
                         -/
  Filter.ext fun x => by simp only [mem_sup, mem_join]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem iSup_join {ι : Sort w} {f : ι → Filter (Filter α)} : ⨆ x, join (f x) = join (⨆ x, f x) :=
                         /-
                           α : Type u
                           ι : Sort w
                           f : ι → Filter (Filter α)
                           x : Set α
                           ⊢ Iff (Membership.mem (iSup fun x => (f x).join) x) (Membership.mem (iSup fun  …
                         -/
  Filter.ext fun x => by simp only [mem_iSup, mem_join]
                         /-
                           🎉 no goals
                         -/


instance : DistribLattice (Filter α) :=
  { Filter.instCompleteLatticeFilter with
    le_sup_inf := by
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        ι : Sort x
        f g : Filter α
        s t : Set α
        ⊢ ∀ (x y z : Filter α), LE.le (Min.min (Max.max x y) (Max.max x z)) (Max.max x …
      -/
      intro x y z s
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        ι : Sort x
        f g : Filter α
        s✝ t : Set α
        x y z : Filter α
        s : Set α
        ⊢ Membership.mem (Max.max x (Min.min y z)) s → Membership.mem (Min.min (Max.ma …
      -/
      simp only [and_assoc, mem_inf_iff, mem_sup, exists_prop, exists_imp, and_imp]
      /-
        α : Type u
        β : Type v
        γ : Type w
        δ : Type u_1
        ι : Sort x
        f g : Filter α
        s✝ t : Set α
        x y z : Filter α
        s : Set α
        ⊢ Membership.mem x s → ∀ (x_1 : Set α), Membership.mem y x_1 → ∀ (x_2 : Set α) …
      -/
      rintro hs t₁ ht₁ t₂ ht₂ rfl
      exact
        ⟨t₁, x.sets_of_superset hs inter_subset_left, ht₁, t₂,
          x.sets_of_superset hs inter_subset_right, ht₂, rfl⟩ }


/-- If `f : ι → Filter α` is directed, `ι` is not empty, and `∀ i, f i ≠ ⊥`, then `iInf f ≠ ⊥`.
See also `iInf_neBot_of_directed` for a version assuming `Nonempty α` instead of `Nonempty ι`. -/
theorem iInf_neBot_of_directed' {f : ι → Filter α} [Nonempty ι] (hd : Directed (· ≥ ·) f) :
    (∀ i, NeBot (f i)) → NeBot (iInf f) :=
  not_imp_not.1 <| by simpa only [not_forall, not_neBot, ← empty_mem_iff_bot,
    mem_iInf_of_directed hd] using id


/-- If `f : ι → Filter α` is directed, `α` is not empty, and `∀ i, f i ≠ ⊥`, then `iInf f ≠ ⊥`.
See also `iInf_neBot_of_directed'` for a version assuming `Nonempty ι` instead of `Nonempty α`. -/
theorem iInf_neBot_of_directed {f : ι → Filter α} [hn : Nonempty α] (hd : Directed (· ≥ ·) f)
    (hb : ∀ i, NeBot (f i)) : NeBot (iInf f) := by
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    hn : Nonempty α
    hd : Directed (fun x1 x2 => GE.ge x1 x2) f
    hb : ∀ (i : ι), (f i).NeBot
    ⊢ (iInf f).NeBot
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      α : Type u
      ι : Sort x
      f : ι → Filter α
      hn : Nonempty α
      hd : Directed (fun x1 x2 => GE.ge x1 x2) f
      hb : ∀ (i : ι), (f i).NeBot
      h✝ : IsEmpty ι
      ⊢ (iInf f).NeBot
    -/
  · constructor
    /-
      case inl.ne'
      α : Type u
      ι : Sort x
      f : ι → Filter α
      hn : Nonempty α
      hd : Directed (fun x1 x2 => GE.ge x1 x2) f
      hb : ∀ (i : ι), (f i).NeBot
      h✝ : IsEmpty ι
      ⊢ Ne (iInf f) Bot.bot
    -/
    simp [iInf_of_empty f, top_ne_bot]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      ι : Sort x
      f : ι → Filter α
      hn : Nonempty α
      hd : Directed (fun x1 x2 => GE.ge x1 x2) f
      hb : ∀ (i : ι), (f i).NeBot
      h✝ : Nonempty ι
      ⊢ (iInf f).NeBot
    -/
  · exact iInf_neBot_of_directed' hd hb
    /-
      🎉 no goals
    -/


theorem sInf_neBot_of_directed' {s : Set (Filter α)} (hne : s.Nonempty) (hd : DirectedOn (· ≥ ·) s)
    (hbot : ⊥ ∉ s) : NeBot (sInf s) :=
  (sInf_eq_iInf' s).symm ▸
    @iInf_neBot_of_directed' _ _ _ hne.to_subtype hd.directed_val fun ⟨_, hf⟩ =>
      ⟨ne_of_mem_of_not_mem hf hbot⟩


theorem sInf_neBot_of_directed [Nonempty α] {s : Set (Filter α)} (hd : DirectedOn (· ≥ ·) s)
    (hbot : ⊥ ∉ s) : NeBot (sInf s) :=
  (sInf_eq_iInf' s).symm ▸
    iInf_neBot_of_directed hd.directed_val fun ⟨_, hf⟩ => ⟨ne_of_mem_of_not_mem hf hbot⟩


theorem iInf_neBot_iff_of_directed' {f : ι → Filter α} [Nonempty ι] (hd : Directed (· ≥ ·) f) :
    NeBot (iInf f) ↔ ∀ i, NeBot (f i) :=
  ⟨fun H i => H.mono (iInf_le _ i), iInf_neBot_of_directed' hd⟩


theorem iInf_neBot_iff_of_directed {f : ι → Filter α} [Nonempty α] (hd : Directed (· ≥ ·) f) :
    NeBot (iInf f) ↔ ∀ i, NeBot (f i) :=
  ⟨fun H i => H.mono (iInf_le _ i), iInf_neBot_of_directed hd⟩


@[simp]
theorem inf_principal {s t : Set α} : 𝓟 s ⊓ 𝓟 t = 𝓟 (s ∩ t) :=
  le_antisymm
        /-
          α : Type u
          s t : Set α
          ⊢ LE.le (Min.min (Filter.principal s) (Filter.principal t)) (Filter.principal  …
        -/
    (by simp only [le_principal_iff, mem_inf_iff]; exact ⟨s, Subset.rfl, t, Subset.rfl, rfl⟩)
                                                   /-
                                                     🎉 no goals
                                                   -/
        /-
          α : Type u
          s t : Set α
          ⊢ LE.le (Filter.principal (Inter.inter s t)) (Min.min (Filter.principal s) (Fi …
        -/
    (by simp [le_inf_iff, inter_subset_left, inter_subset_right])
        /-
          🎉 no goals
        -/


@[simp]
theorem sup_principal {s t : Set α} : 𝓟 s ⊔ 𝓟 t = 𝓟 (s ∪ t) :=
                         /-
                           α : Type u
                           s t u : Set α
                           ⊢ Iff (Membership.mem (Max.max (Filter.principal s) (Filter.principal t)) u) ( …
                         -/
  Filter.ext fun u => by simp only [union_subset_iff, mem_sup, mem_principal]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem iSup_principal {ι : Sort w} {s : ι → Set α} : ⨆ x, 𝓟 (s x) = 𝓟 (⋃ i, s i) :=
                         /-
                           α : Type u
                           ι : Sort w
                           s : ι → Set α
                           x : Set α
                           ⊢ Iff (Membership.mem (iSup fun x => Filter.principal (s x)) x) (Membership.me …
                         -/
  Filter.ext fun x => by simp only [mem_iSup, mem_principal, iUnion_subset_iff]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem principal_eq_bot_iff {s : Set α} : 𝓟 s = ⊥ ↔ s = ∅ :=
  empty_mem_iff_bot.symm.trans <| mem_principal.trans subset_empty_iff


@[simp]
theorem principal_neBot_iff {s : Set α} : NeBot (𝓟 s) ↔ s.Nonempty :=
  neBot_iff.trans <| (not_congr principal_eq_bot_iff).trans nonempty_iff_ne_empty.symm


alias ⟨_, _root_.Set.Nonempty.principal_neBot⟩ := principal_neBot_iff


theorem isCompl_principal (s : Set α) : IsCompl (𝓟 s) (𝓟 sᶜ) :=
                    /-
                      α : Type u
                      s : Set α
                      ⊢ Eq (Min.min (Filter.principal s) (Filter.principal (HasCompl.compl s))) Bot. …
                    -/
  IsCompl.of_eq (by rw [inf_principal, inter_compl_self, principal_empty]) <| by
                    /-
                      🎉 no goals
                    -/
    /-
      α : Type u
      s : Set α
      ⊢ Eq (Max.max (Filter.principal s) (Filter.principal (HasCompl.compl s))) Top. …
    -/
    rw [sup_principal, union_compl_self, principal_univ]
    /-
      🎉 no goals
    -/


theorem mem_inf_principal' {f : Filter α} {s t : Set α} : s ∈ f ⊓ 𝓟 t ↔ tᶜ ∪ s ∈ f := by
  simp only [← le_principal_iff, (isCompl_principal s).le_left_iff, disjoint_assoc, inf_principal,
    ← (isCompl_principal (t ∩ sᶜ)).le_right_iff, compl_inter, compl_compl]


lemma mem_inf_principal {f : Filter α} {s t : Set α} : s ∈ f ⊓ 𝓟 t ↔ { x | x ∈ t → x ∈ s } ∈ f := by
  /-
    α : Type u
    f : Filter α
    s t : Set α
    ⊢ Iff (Membership.mem (Min.min f (Filter.principal t)) s) (Membership.mem f (s …
  -/
  simp only [mem_inf_principal', imp_iff_not_or, setOf_or, compl_def, setOf_mem_eq]
  /-
    🎉 no goals
  -/


lemma iSup_inf_principal (f : ι → Filter α) (s : Set α) : ⨆ i, f i ⊓ 𝓟 s = (⨆ i, f i) ⊓ 𝓟 s := by
  /-
    α : Type u
    ι : Sort x
    f : ι → Filter α
    s : Set α
    ⊢ Eq (iSup fun i => Min.min (f i) (Filter.principal s)) (Min.min (iSup fun i = …
  -/
  ext
  /-
    case h
    α : Type u
    ι : Sort x
    f : ι → Filter α
    s s✝ : Set α
    ⊢ Iff (Membership.mem (iSup fun i => Min.min (f i) (Filter.principal s)) s✝) ( …
  -/
  simp only [mem_iSup, mem_inf_principal]
  /-
    🎉 no goals
  -/


theorem inf_principal_eq_bot {f : Filter α} {s : Set α} : f ⊓ 𝓟 s = ⊥ ↔ sᶜ ∈ f := by
  /-
    α : Type u
    f : Filter α
    s : Set α
    ⊢ Iff (Eq (Min.min f (Filter.principal s)) Bot.bot) (Membership.mem f (HasComp …
  -/
  rw [← empty_mem_iff_bot, mem_inf_principal]
  /-
    α : Type u
    f : Filter α
    s : Set α
    ⊢ Iff (Membership.mem f (setOf fun x => Membership.mem s x → Membership.mem Em …
  -/
  simp only [mem_empty_iff_false, imp_false, compl_def]
  /-
    🎉 no goals
  -/


theorem mem_of_eq_bot {f : Filter α} {s : Set α} (h : f ⊓ 𝓟 sᶜ = ⊥) : s ∈ f := by
  /-
    α : Type u
    f : Filter α
    s : Set α
    h : Eq (Min.min f (Filter.principal (HasCompl.compl s))) Bot.bot
    ⊢ Membership.mem f s
  -/
  rwa [inf_principal_eq_bot, compl_compl] at h
  /-
    🎉 no goals
  -/


theorem diff_mem_inf_principal_compl {f : Filter α} {s : Set α} (hs : s ∈ f) (t : Set α) :
    s \ t ∈ f ⊓ 𝓟 tᶜ :=
  inter_mem_inf hs <| mem_principal_self tᶜ


theorem principal_le_iff {s : Set α} {f : Filter α} : 𝓟 s ≤ f ↔ ∀ V ∈ f, s ⊆ V := by
  /-
    α : Type u
    s : Set α
    f : Filter α
    ⊢ Iff (LE.le (Filter.principal s) f) (∀ (V : Set α), Membership.mem f V → HasS …
  -/
  simp_rw [le_def, mem_principal]
  /-
    🎉 no goals
  -/


@[mono, gcongr]
theorem join_mono {f₁ f₂ : Filter (Filter α)} (h : f₁ ≤ f₂) : join f₁ ≤ join f₂ := fun _ hs => h hs


theorem eventually_iff {f : Filter α} {P : α → Prop} : (∀ᶠ x in f, P x) ↔ { x | P x } ∈ f :=
  Iff.rfl


@[simp]
theorem eventually_mem_set {s : Set α} {l : Filter α} : (∀ᶠ x in l, x ∈ s) ↔ s ∈ l :=
  Iff.rfl


protected theorem ext' {f₁ f₂ : Filter α}
    (h : ∀ p : α → Prop, (∀ᶠ x in f₁, p x) ↔ ∀ᶠ x in f₂, p x) : f₁ = f₂ :=
  Filter.ext h


theorem Eventually.filter_mono {f₁ f₂ : Filter α} (h : f₁ ≤ f₂) {p : α → Prop}
    (hp : ∀ᶠ x in f₂, p x) : ∀ᶠ x in f₁, p x :=
  h hp


theorem eventually_of_mem {f : Filter α} {P : α → Prop} {U : Set α} (hU : U ∈ f)
    (h : ∀ x ∈ U, P x) : ∀ᶠ x in f, P x :=
  mem_of_superset hU h


protected theorem Eventually.and {p q : α → Prop} {f : Filter α} :
    f.Eventually p → f.Eventually q → ∀ᶠ x in f, p x ∧ q x :=
  inter_mem


@[simp] theorem eventually_true (f : Filter α) : ∀ᶠ _ in f, True := univ_mem


theorem Eventually.of_forall {p : α → Prop} {f : Filter α} (hp : ∀ x, p x) : ∀ᶠ x in f, p x :=
  univ_mem' hp


@[deprecated (since := "2024-08-02")] alias eventually_of_forall := Eventually.of_forall


@[simp]
theorem eventually_false_iff_eq_bot {f : Filter α} : (∀ᶠ _ in f, False) ↔ f = ⊥ :=
  empty_mem_iff_bot


@[simp]
theorem eventually_const {f : Filter α} [t : NeBot f] {p : Prop} : (∀ᶠ _ in f, p) ↔ p := by
  /-
    α : Type u
    f : Filter α
    t : f.NeBot
    p : Prop
    ⊢ Iff (Filter.Eventually (fun x => p) f) p
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : p <;> simp [h, t.ne]
                     /-
                       🎉 no goals
                     -/


theorem eventually_iff_exists_mem {p : α → Prop} {f : Filter α} :
    (∀ᶠ x in f, p x) ↔ ∃ v ∈ f, ∀ y ∈ v, p y :=
  exists_mem_subset_iff.symm


theorem Eventually.exists_mem {p : α → Prop} {f : Filter α} (hp : ∀ᶠ x in f, p x) :
    ∃ v ∈ f, ∀ y ∈ v, p y :=
  eventually_iff_exists_mem.1 hp


theorem Eventually.mp {p q : α → Prop} {f : Filter α} (hp : ∀ᶠ x in f, p x)
    (hq : ∀ᶠ x in f, p x → q x) : ∀ᶠ x in f, q x :=
  mp_mem hp hq


theorem Eventually.mono {p q : α → Prop} {f : Filter α} (hp : ∀ᶠ x in f, p x)
    (hq : ∀ x, p x → q x) : ∀ᶠ x in f, q x :=
  hp.mp (Eventually.of_forall hq)


theorem forall_eventually_of_eventually_forall {f : Filter α} {p : α → β → Prop}
    (h : ∀ᶠ x in f, ∀ y, p x y) : ∀ y, ∀ᶠ x in f, p x y :=
  fun y => h.mono fun _ h => h y


@[simp]
theorem eventually_and {p q : α → Prop} {f : Filter α} :
    (∀ᶠ x in f, p x ∧ q x) ↔ (∀ᶠ x in f, p x) ∧ ∀ᶠ x in f, q x :=
  inter_mem_iff


theorem Eventually.congr {f : Filter α} {p q : α → Prop} (h' : ∀ᶠ x in f, p x)
    (h : ∀ᶠ x in f, p x ↔ q x) : ∀ᶠ x in f, q x :=
  h'.mp (h.mono fun _ hx => hx.mp)


theorem eventually_congr {f : Filter α} {p q : α → Prop} (h : ∀ᶠ x in f, p x ↔ q x) :
    (∀ᶠ x in f, p x) ↔ ∀ᶠ x in f, q x :=
                                                  /-
                                                    α : Type u
                                                    f : Filter α
                                                    p q : α → Prop
                                                    h : Filter.Eventually (fun x => Iff (p x) (q x)) f
                                                    hq : Filter.Eventually (fun x => q x) f
                                                    ⊢ Filter.Eventually (fun x => Iff (q x) (p x)) f
                                                  -/
  ⟨fun hp => hp.congr h, fun hq => hq.congr <| by simpa only [Iff.comm] using h⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem eventually_or_distrib_left {f : Filter α} {p : Prop} {q : α → Prop} :
    (∀ᶠ x in f, p ∨ q x) ↔ p ∨ ∀ᶠ x in f, q x :=
                            /-
                              α : Type u
                              f : Filter α
                              p : Prop
                              q : α → Prop
                              h : p
                              ⊢ Iff (Filter.Eventually (fun x => Or p (q x)) f) (Or p (Filter.Eventually (fu …
                            -/
                            /-
                              🎉 no goals
                            -/
  by_cases (fun h : p => by simp [h]) fun h => by simp [h]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem eventually_or_distrib_right {f : Filter α} {p : α → Prop} {q : Prop} :
    (∀ᶠ x in f, p x ∨ q) ↔ (∀ᶠ x in f, p x) ∨ q := by
  /-
    α : Type u
    f : Filter α
    p : α → Prop
    q : Prop
    ⊢ Iff (Filter.Eventually (fun x => Or (p x) q) f) (Or (Filter.Eventually (fun  …
  -/
  simp only [@or_comm _ q, eventually_or_distrib_left]
  /-
    🎉 no goals
  -/


theorem eventually_imp_distrib_left {f : Filter α} {p : Prop} {q : α → Prop} :
    (∀ᶠ x in f, p → q x) ↔ p → ∀ᶠ x in f, q x := by
  /-
    α : Type u
    f : Filter α
    p : Prop
    q : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p → q x) f) (p → Filter.Eventually (fun x = …
  -/
  simp only [imp_iff_not_or, eventually_or_distrib_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem eventually_bot {p : α → Prop} : ∀ᶠ x in ⊥, p x :=
  ⟨⟩


@[simp]
theorem eventually_top {p : α → Prop} : (∀ᶠ x in ⊤, p x) ↔ ∀ x, p x :=
  Iff.rfl


@[simp]
theorem eventually_sup {p : α → Prop} {f g : Filter α} :
    (∀ᶠ x in f ⊔ g, p x) ↔ (∀ᶠ x in f, p x) ∧ ∀ᶠ x in g, p x :=
  Iff.rfl


@[simp]
theorem eventually_sSup {p : α → Prop} {fs : Set (Filter α)} :
    (∀ᶠ x in sSup fs, p x) ↔ ∀ f ∈ fs, ∀ᶠ x in f, p x :=
  Iff.rfl


@[simp]
theorem eventually_iSup {p : α → Prop} {fs : ι → Filter α} :
    (∀ᶠ x in ⨆ b, fs b, p x) ↔ ∀ b, ∀ᶠ x in fs b, p x :=
  mem_iSup


@[simp]
theorem eventually_principal {a : Set α} {p : α → Prop} : (∀ᶠ x in 𝓟 a, p x) ↔ ∀ x ∈ a, p x :=
  Iff.rfl


theorem Eventually.forall_mem {α : Type*} {f : Filter α} {s : Set α} {P : α → Prop}
    (hP : ∀ᶠ x in f, P x) (hf : 𝓟 s ≤ f) : ∀ x ∈ s, P x :=
  Filter.eventually_principal.mp (hP.filter_mono hf)


theorem eventually_inf {f g : Filter α} {p : α → Prop} :
    (∀ᶠ x in f ⊓ g, p x) ↔ ∃ s ∈ f, ∃ t ∈ g, ∀ x ∈ s ∩ t, p x :=
  mem_inf_iff_superset


theorem eventually_inf_principal {f : Filter α} {p : α → Prop} {s : Set α} :
    (∀ᶠ x in f ⊓ 𝓟 s, p x) ↔ ∀ᶠ x in f, x ∈ s → p x :=
  mem_inf_principal


theorem eventually_iff_all_subsets {f : Filter α} {p : α → Prop} :
    (∀ᶠ x in f, p x) ↔ ∀ (s : Set α), ∀ᶠ x in f, x ∈ s → p x where
               /-
                 α : Type u
                 f : Filter α
                 p : α → Prop
                 h : Filter.Eventually (fun x => p x) f
                 x✝ : Set α
                 ⊢ Filter.Eventually (fun x => Membership.mem x✝ x → p x) f
               -/
  mp h _ := by filter_upwards [h] with _ pa _ using pa
               /-
                 🎉 no goals
               -/
              /-
                α : Type u
                f : Filter α
                p : α → Prop
                h : ∀ (s : Set α), Filter.Eventually (fun x => Membership.mem s x → p x) f
                ⊢ Filter.Eventually (fun x => p x) f
              -/
  mpr h := by filter_upwards [h univ] with _ pa using pa (by simp)
              /-
                🎉 no goals
              -/


theorem Eventually.frequently {f : Filter α} [NeBot f] {p : α → Prop} (h : ∀ᶠ x in f, p x) :
    ∃ᶠ x in f, p x :=
  compl_not_mem h


theorem Frequently.of_forall {f : Filter α} [NeBot f] {p : α → Prop} (h : ∀ x, p x) :
    ∃ᶠ x in f, p x :=
  Eventually.frequently (Eventually.of_forall h)


@[deprecated (since := "2024-08-02")] alias frequently_of_forall := Frequently.of_forall


theorem Frequently.mp {p q : α → Prop} {f : Filter α} (h : ∃ᶠ x in f, p x)
    (hpq : ∀ᶠ x in f, p x → q x) : ∃ᶠ x in f, q x :=
  mt (fun hq => hq.mp <| hpq.mono fun _ => mt) h


lemma frequently_congr {p q : α → Prop} {f : Filter α} (h : ∀ᶠ x in f, p x ↔ q x) :
    (∃ᶠ x in f, p x) ↔ ∃ᶠ x in f, q x :=
  ⟨fun h' ↦ h'.mp (h.mono fun _ ↦ Iff.mp), fun h' ↦ h'.mp (h.mono fun _ ↦ Iff.mpr)⟩


theorem Frequently.filter_mono {p : α → Prop} {f g : Filter α} (h : ∃ᶠ x in f, p x) (hle : f ≤ g) :
    ∃ᶠ x in g, p x :=
  mt (fun h' => h'.filter_mono hle) h


theorem Frequently.mono {p q : α → Prop} {f : Filter α} (h : ∃ᶠ x in f, p x)
    (hpq : ∀ x, p x → q x) : ∃ᶠ x in f, q x :=
  h.mp (Eventually.of_forall hpq)


theorem Frequently.and_eventually {p q : α → Prop} {f : Filter α} (hp : ∃ᶠ x in f, p x)
    (hq : ∀ᶠ x in f, q x) : ∃ᶠ x in f, p x ∧ q x := by
  /-
    α : Type u
    p q : α → Prop
    f : Filter α
    hp : Filter.Frequently (fun x => p x) f
    hq : Filter.Eventually (fun x => q x) f
    ⊢ Filter.Frequently (fun x => And (p x) (q x)) f
  -/
  refine mt (fun h => hq.mp <| h.mono ?_) hp
  /-
    α : Type u
    p q : α → Prop
    f : Filter α
    hp : Filter.Frequently (fun x => p x) f
    hq : Filter.Eventually (fun x => q x) f
    h : Filter.Eventually (fun x => Not ((fun x => And (p x) (q x)) x)) f
    ⊢ ∀ (x : α), Not ((fun x => And (p x) (q x)) x) → q x → Not ((fun x => p x) x)
  -/
  exact fun x hpq hq hp => hpq ⟨hp, hq⟩
  /-
    🎉 no goals
  -/


theorem Eventually.and_frequently {p q : α → Prop} {f : Filter α} (hp : ∀ᶠ x in f, p x)
    (hq : ∃ᶠ x in f, q x) : ∃ᶠ x in f, p x ∧ q x := by
  /-
    α : Type u
    p q : α → Prop
    f : Filter α
    hp : Filter.Eventually (fun x => p x) f
    hq : Filter.Frequently (fun x => q x) f
    ⊢ Filter.Frequently (fun x => And (p x) (q x)) f
  -/
  simpa only [and_comm] using hq.and_eventually hp
  /-
    🎉 no goals
  -/


theorem Frequently.exists {p : α → Prop} {f : Filter α} (hp : ∃ᶠ x in f, p x) : ∃ x, p x := by
  /-
    α : Type u
    p : α → Prop
    f : Filter α
    hp : Filter.Frequently (fun x => p x) f
    ⊢ Exists fun x => p x
  -/
  by_contra H
  /-
    α : Type u
    p : α → Prop
    f : Filter α
    hp : Filter.Frequently (fun x => p x) f
    H : Not (Exists fun x => p x)
    ⊢ False
  -/
  replace H : ∀ᶠ x in f, ¬p x := Eventually.of_forall (not_exists.1 H)
  /-
    α : Type u
    p : α → Prop
    f : Filter α
    hp : Filter.Frequently (fun x => p x) f
    H : Filter.Eventually (fun x => Not (p x)) f
    ⊢ False
  -/
  exact hp H
  /-
    🎉 no goals
  -/


theorem Eventually.exists {p : α → Prop} {f : Filter α} [NeBot f] (hp : ∀ᶠ x in f, p x) :
    ∃ x, p x :=
  hp.frequently.exists


lemma frequently_iff_neBot {l : Filter α} {p : α → Prop} :
    (∃ᶠ x in l, p x) ↔ NeBot (l ⊓ 𝓟 {x | p x}) := by
  /-
    α : Type u
    l : Filter α
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x) l) (Min.min l (Filter.principal (setOf …
  -/
  rw [neBot_iff, Ne, inf_principal_eq_bot]; rfl
                                            /-
                                              🎉 no goals
                                            -/


lemma frequently_mem_iff_neBot {l : Filter α} {s : Set α} : (∃ᶠ x in l, x ∈ s) ↔ NeBot (l ⊓ 𝓟 s) :=
  frequently_iff_neBot


theorem frequently_iff_forall_eventually_exists_and {p : α → Prop} {f : Filter α} :
    (∃ᶠ x in f, p x) ↔ ∀ {q : α → Prop}, (∀ᶠ x in f, q x) → ∃ x, p x ∧ q x :=
  ⟨fun hp _ hq => (hp.and_eventually hq).exists, fun H hp => by
    /-
      α : Type u
      p : α → Prop
      f : Filter α
      H : ∀ {q : α → Prop}, Filter.Eventually (fun x => q x) f → Exists fun x => And …
      hp : Filter.Eventually (fun x => Not ((fun x => p x) x)) f
      ⊢ False
    -/
    simpa only [and_not_self_iff, exists_false] using H hp⟩
    /-
      🎉 no goals
    -/


theorem frequently_iff {f : Filter α} {P : α → Prop} :
    (∃ᶠ x in f, P x) ↔ ∀ {U}, U ∈ f → ∃ x ∈ U, P x := by
  /-
    α : Type u
    f : Filter α
    P : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => P x) f) (∀ {U : Set α}, Membership.mem f U  …
  -/
  simp only [frequently_iff_forall_eventually_exists_and, @and_comm (P _)]
  /-
    α : Type u
    f : Filter α
    P : α → Prop
    ⊢ Iff (∀ {q : α → Prop}, Filter.Eventually (fun x => q x) f → Exists fun x =>  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem not_eventually {p : α → Prop} {f : Filter α} : (¬∀ᶠ x in f, p x) ↔ ∃ᶠ x in f, ¬p x := by
  /-
    α : Type u
    p : α → Prop
    f : Filter α
    ⊢ Iff (Not (Filter.Eventually (fun x => p x) f)) (Filter.Frequently (fun x =>  …
  -/
  simp [Filter.Frequently]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_frequently {p : α → Prop} {f : Filter α} : (¬∃ᶠ x in f, p x) ↔ ∀ᶠ x in f, ¬p x := by
  /-
    α : Type u
    p : α → Prop
    f : Filter α
    ⊢ Iff (Not (Filter.Frequently (fun x => p x) f)) (Filter.Eventually (fun x =>  …
  -/
  simp only [Filter.Frequently, not_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem frequently_true_iff_neBot (f : Filter α) : (∃ᶠ _ in f, True) ↔ NeBot f := by
  /-
    α : Type u
    f : Filter α
    ⊢ Iff (Filter.Frequently (fun x => True) f) f.NeBot
  -/
  simp [frequently_iff_neBot]
  /-
    🎉 no goals
  -/


@[simp]
                                                                  /-
                                                                    α : Type u
                                                                    f : Filter α
                                                                    ⊢ Not (Filter.Frequently (fun x => False) f)
                                                                  -/
theorem frequently_false (f : Filter α) : ¬∃ᶠ _ in f, False := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem frequently_const {f : Filter α} [NeBot f] {p : Prop} : (∃ᶠ _ in f, p) ↔ p := by
  /-
    α : Type u
    f : Filter α
    inst✝ : f.NeBot
    p : Prop
    ⊢ Iff (Filter.Frequently (fun x => p) f) p
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases p <;> simp [*]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem frequently_or_distrib {f : Filter α} {p q : α → Prop} :
    (∃ᶠ x in f, p x ∨ q x) ↔ (∃ᶠ x in f, p x) ∨ ∃ᶠ x in f, q x := by
  /-
    α : Type u
    f : Filter α
    p q : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => Or (p x) (q x)) f) (Or (Filter.Frequently ( …
  -/
  simp only [Filter.Frequently, ← not_and_or, not_or, eventually_and]
  /-
    🎉 no goals
  -/


theorem frequently_or_distrib_left {f : Filter α} [NeBot f] {p : Prop} {q : α → Prop} :
                                                    /-
                                                      α : Type u
                                                      f : Filter α
                                                      inst✝ : f.NeBot
                                                      p : Prop
                                                      q : α → Prop
                                                      ⊢ Iff (Filter.Frequently (fun x => Or p (q x)) f) (Or p (Filter.Frequently (fu …
                                                    -/
    (∃ᶠ x in f, p ∨ q x) ↔ p ∨ ∃ᶠ x in f, q x := by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem frequently_or_distrib_right {f : Filter α} [NeBot f] {p : α → Prop} {q : Prop} :
                                                      /-
                                                        α : Type u
                                                        f : Filter α
                                                        inst✝ : f.NeBot
                                                        p : α → Prop
                                                        q : Prop
                                                        ⊢ Iff (Filter.Frequently (fun x => Or (p x) q) f) (Or (Filter.Frequently (fun  …
                                                      -/
    (∃ᶠ x in f, p x ∨ q) ↔ (∃ᶠ x in f, p x) ∨ q := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem frequently_imp_distrib {f : Filter α} {p q : α → Prop} :
    (∃ᶠ x in f, p x → q x) ↔ (∀ᶠ x in f, p x) → ∃ᶠ x in f, q x := by
  /-
    α : Type u
    f : Filter α
    p q : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x → q x) f) (Filter.Eventually (fun x =>  …
  -/
  simp [imp_iff_not_or]
  /-
    🎉 no goals
  -/


theorem frequently_imp_distrib_left {f : Filter α} [NeBot f] {p : Prop} {q : α → Prop} :
                                                    /-
                                                      α : Type u
                                                      f : Filter α
                                                      inst✝ : f.NeBot
                                                      p : Prop
                                                      q : α → Prop
                                                      ⊢ Iff (Filter.Frequently (fun x => p → q x) f) (p → Filter.Frequently (fun x = …
                                                    -/
    (∃ᶠ x in f, p → q x) ↔ p → ∃ᶠ x in f, q x := by simp [frequently_imp_distrib]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem frequently_imp_distrib_right {f : Filter α} [NeBot f] {p : α → Prop} {q : Prop} :
    (∃ᶠ x in f, p x → q) ↔ (∀ᶠ x in f, p x) → q := by
  /-
    α : Type u
    f : Filter α
    inst✝ : f.NeBot
    p : α → Prop
    q : Prop
    ⊢ Iff (Filter.Frequently (fun x => p x → q) f) (Filter.Eventually (fun x => p  …
  -/
  simp only [frequently_imp_distrib, frequently_const]
  /-
    🎉 no goals
  -/


theorem eventually_imp_distrib_right {f : Filter α} {p : α → Prop} {q : Prop} :
    (∀ᶠ x in f, p x → q) ↔ (∃ᶠ x in f, p x) → q := by
  /-
    α : Type u
    f : Filter α
    p : α → Prop
    q : Prop
    ⊢ Iff (Filter.Eventually (fun x => p x → q) f) (Filter.Frequently (fun x => p  …
  -/
  simp only [imp_iff_not_or, eventually_or_distrib_right, not_frequently]
  /-
    🎉 no goals
  -/


@[simp]
theorem frequently_and_distrib_left {f : Filter α} {p : Prop} {q : α → Prop} :
    (∃ᶠ x in f, p ∧ q x) ↔ p ∧ ∃ᶠ x in f, q x := by
  /-
    α : Type u
    f : Filter α
    p : Prop
    q : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => And p (q x)) f) (And p (Filter.Frequently ( …
  -/
  simp only [Filter.Frequently, not_and, eventually_imp_distrib_left, Classical.not_imp]
  /-
    🎉 no goals
  -/


@[simp]
theorem frequently_and_distrib_right {f : Filter α} {p : α → Prop} {q : Prop} :
    (∃ᶠ x in f, p x ∧ q) ↔ (∃ᶠ x in f, p x) ∧ q := by
  /-
    α : Type u
    f : Filter α
    p : α → Prop
    q : Prop
    ⊢ Iff (Filter.Frequently (fun x => And (p x) q) f) (And (Filter.Frequently (fu …
  -/
  simp only [@and_comm _ q, frequently_and_distrib_left]
  /-
    🎉 no goals
  -/


@[simp]
                                                              /-
                                                                α : Type u
                                                                p : α → Prop
                                                                ⊢ Not (Filter.Frequently (fun x => p x) Bot.bot)
                                                              -/
theorem frequently_bot {p : α → Prop} : ¬∃ᶠ x in ⊥, p x := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                          /-
                                                                            α : Type u
                                                                            p : α → Prop
                                                                            ⊢ Iff (Filter.Frequently (fun x => p x) Top.top) (Exists fun x => p x)
                                                                          -/
theorem frequently_top {p : α → Prop} : (∃ᶠ x in ⊤, p x) ↔ ∃ x, p x := by simp [Filter.Frequently]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem frequently_principal {a : Set α} {p : α → Prop} : (∃ᶠ x in 𝓟 a, p x) ↔ ∃ x ∈ a, p x := by
  /-
    α : Type u
    a : Set α
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x) (Filter.principal a)) (Exists fun x => …
  -/
  simp [Filter.Frequently, not_forall]
  /-
    🎉 no goals
  -/


theorem frequently_inf_principal {f : Filter α} {s : Set α} {p : α → Prop} :
    (∃ᶠ x in f ⊓ 𝓟 s, p x) ↔ ∃ᶠ x in f, x ∈ s ∧ p x := by
  /-
    α : Type u
    f : Filter α
    s : Set α
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x) (Min.min f (Filter.principal s))) (Fil …
  -/
  simp only [Filter.Frequently, eventually_inf_principal, not_and]
  /-
    🎉 no goals
  -/


alias ⟨Frequently.of_inf_principal, Frequently.inf_principal⟩ := frequently_inf_principal


theorem frequently_sup {p : α → Prop} {f g : Filter α} :
    (∃ᶠ x in f ⊔ g, p x) ↔ (∃ᶠ x in f, p x) ∨ ∃ᶠ x in g, p x := by
  /-
    α : Type u
    p : α → Prop
    f g : Filter α
    ⊢ Iff (Filter.Frequently (fun x => p x) (Max.max f g)) (Or (Filter.Frequently  …
  -/
  simp only [Filter.Frequently, eventually_sup, not_and_or]
  /-
    🎉 no goals
  -/


@[simp]
theorem frequently_sSup {p : α → Prop} {fs : Set (Filter α)} :
    (∃ᶠ x in sSup fs, p x) ↔ ∃ f ∈ fs, ∃ᶠ x in f, p x := by
  /-
    α : Type u
    p : α → Prop
    fs : Set (Filter α)
    ⊢ Iff (Filter.Frequently (fun x => p x) (SupSet.sSup fs)) (Exists fun f => And …
  -/
  simp only [Filter.Frequently, not_forall, eventually_sSup, exists_prop]
  /-
    🎉 no goals
  -/


@[simp]
theorem frequently_iSup {p : α → Prop} {fs : β → Filter α} :
    (∃ᶠ x in ⨆ b, fs b, p x) ↔ ∃ b, ∃ᶠ x in fs b, p x := by
  /-
    α : Type u
    β : Type v
    p : α → Prop
    fs : β → Filter α
    ⊢ Iff (Filter.Frequently (fun x => p x) (iSup fun b => fs b)) (Exists fun b => …
  -/
  simp only [Filter.Frequently, eventually_iSup, not_forall]
  /-
    🎉 no goals
  -/


theorem Eventually.choice {r : α → β → Prop} {l : Filter α} [l.NeBot] (h : ∀ᶠ x in l, ∃ y, r x y) :
    ∃ f : α → β, ∀ᶠ x in l, r x (f x) := by
  /-
    α : Type u
    β : Type v
    r : α → β → Prop
    l : Filter α
    inst✝ : l.NeBot
    h : Filter.Eventually (fun x => Exists fun y => r x y) l
    ⊢ Exists fun f => Filter.Eventually (fun x => r x (f x)) l
  -/
  haveI : Nonempty β := let ⟨_, hx⟩ := h.exists; hx.nonempty
  /-
    α : Type u
    β : Type v
    r : α → β → Prop
    l : Filter α
    inst✝ : l.NeBot
    h : Filter.Eventually (fun x => Exists fun y => r x y) l
    this : Nonempty β
    ⊢ Exists fun f => Filter.Eventually (fun x => r x (f x)) l
  -/
  choose! f hf using fun x (hx : ∃ y, r x y) => hx
  /-
    α : Type u
    β : Type v
    r : α → β → Prop
    l : Filter α
    inst✝ : l.NeBot
    h : Filter.Eventually (fun x => Exists fun y => r x y) l
    this : Nonempty β
    f : α → β
    hf : ∀ (x : α), (Exists fun y => r x y) → r x (f x)
    ⊢ Exists fun f => Filter.Eventually (fun x => r x (f x)) l
  -/
  exact ⟨f, h.mono hf⟩
  /-
    🎉 no goals
  -/


lemma skolem {ι : Type*} {α : ι → Type*} [∀ i, Nonempty (α i)]
    {P : ∀ i : ι, α i → Prop} {F : Filter ι} :
    (∀ᶠ i in F, ∃ b, P i b) ↔ ∃ b : (Π i, α i), ∀ᶠ i in F, P i (b i) := by
  classical
  refine ⟨fun H ↦ ?_, fun ⟨b, hb⟩ ↦ hb.mp (.of_forall fun x a ↦ ⟨_, a⟩)⟩
  refine ⟨fun i ↦ if h : ∃ b, P i b then h.choose else Nonempty.some inferInstance, ?_⟩
  filter_upwards [H] with i hi
  exact dif_pos hi ▸ hi.choose_spec


theorem EventuallyEq.eventually (h : f =ᶠ[l] g) : ∀ᶠ x in l, f x = g x := h


                                                         /-
                                                           α : Type u
                                                           β : Type v
                                                           f g : α → β
                                                           ⊢ Iff (Top.top.EventuallyEq f g) (Eq f g)
                                                         -/
@[simp] lemma eventuallyEq_top : f =ᶠ[⊤] g ↔ f = g := by simp [EventuallyEq, funext_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem EventuallyEq.rw {l : Filter α} {f g : α → β} (h : f =ᶠ[l] g) (p : α → β → Prop)
    (hf : ∀ᶠ x in l, p x (f x)) : ∀ᶠ x in l, p x (g x) :=
  hf.congr <| h.mono fun _ hx => hx ▸ Iff.rfl


theorem eventuallyEq_set {s t : Set α} {l : Filter α} : s =ᶠ[l] t ↔ ∀ᶠ x in l, x ∈ s ↔ x ∈ t :=
  eventually_congr <| Eventually.of_forall fun _ ↦ eq_iff_iff


alias ⟨EventuallyEq.mem_iff, Eventually.set_eq⟩ := eventuallyEq_set


@[simp]
theorem eventuallyEq_univ {s : Set α} {l : Filter α} : s =ᶠ[l] univ ↔ s ∈ l := by
  /-
    α : Type u
    s : Set α
    l : Filter α
    ⊢ Iff (l.EventuallyEq s Set.univ) (Membership.mem l s)
  -/
  simp [eventuallyEq_set]
  /-
    🎉 no goals
  -/


theorem EventuallyEq.exists_mem {l : Filter α} {f g : α → β} (h : f =ᶠ[l] g) :
    ∃ s ∈ l, EqOn f g s :=
  Eventually.exists_mem h


theorem eventuallyEq_of_mem {l : Filter α} {f g : α → β} {s : Set α} (hs : s ∈ l) (h : EqOn f g s) :
    f =ᶠ[l] g :=
  eventually_of_mem hs h


theorem eventuallyEq_iff_exists_mem {l : Filter α} {f g : α → β} :
    f =ᶠ[l] g ↔ ∃ s ∈ l, EqOn f g s :=
  eventually_iff_exists_mem


theorem EventuallyEq.filter_mono {l l' : Filter α} {f g : α → β} (h₁ : f =ᶠ[l] g) (h₂ : l' ≤ l) :
    f =ᶠ[l'] g :=
  h₂ h₁


@[refl, simp]
theorem EventuallyEq.refl (l : Filter α) (f : α → β) : f =ᶠ[l] f :=
  Eventually.of_forall fun _ => rfl


protected theorem EventuallyEq.rfl {l : Filter α} {f : α → β} : f =ᶠ[l] f :=
  EventuallyEq.refl l f


theorem EventuallyEq.of_eq {l : Filter α} {f g : α → β} (h : f = g) : f =ᶠ[l] g := h ▸ .rfl

alias _root_.Eq.eventuallyEq := EventuallyEq.of_eq


@[symm]
theorem EventuallyEq.symm {f g : α → β} {l : Filter α} (H : f =ᶠ[l] g) : g =ᶠ[l] f :=
  H.mono fun _ => Eq.symm


lemma eventuallyEq_comm {f g : α → β} {l : Filter α} : f =ᶠ[l] g ↔ g =ᶠ[l] f := ⟨.symm, .symm⟩


@[trans]
theorem EventuallyEq.trans {l : Filter α} {f g h : α → β} (H₁ : f =ᶠ[l] g) (H₂ : g =ᶠ[l] h) :
    f =ᶠ[l] h :=
  H₂.rw (fun x y => f x = y) H₁


theorem EventuallyEq.congr_left {l : Filter α} {f g h : α → β} (H : f =ᶠ[l] g) :
    f =ᶠ[l] h ↔ g =ᶠ[l] h :=
  ⟨H.symm.trans, H.trans⟩


theorem EventuallyEq.congr_right {l : Filter α} {f g h : α → β} (H : g =ᶠ[l] h) :
    f =ᶠ[l] g ↔ f =ᶠ[l] h :=
  ⟨(·.trans H), (·.trans H.symm)⟩


instance {l : Filter α} :
    Trans ((· =ᶠ[l] ·) : (α → β) → (α → β) → Prop) (· =ᶠ[l] ·) (· =ᶠ[l] ·) where
  trans := EventuallyEq.trans


theorem EventuallyEq.prod_mk {l} {f f' : α → β} (hf : f =ᶠ[l] f') {g g' : α → γ} (hg : g =ᶠ[l] g') :
    (fun x => (f x, g x)) =ᶠ[l] fun x => (f' x, g' x) :=
  hf.mp <|
    hg.mono <| by
      /-
        α : Type u
        β : Type v
        γ : Type w
        l : Filter α
        f f' : α → β
        hf : l.EventuallyEq f f'
        g g' : α → γ
        hg : l.EventuallyEq g g'
        ⊢ ∀ (x : α), Eq (g x) (g' x) → Eq (f x) (f' x) → Eq ((fun x => { fst := f x, s …
      -/
      intros
      /-
        α : Type u
        β : Type v
        γ : Type w
        l : Filter α
        f f' : α → β
        hf : l.EventuallyEq f f'
        g g' : α → γ
        hg : l.EventuallyEq g g'
        x✝ : α
        a✝¹ : Eq (g x✝) (g' x✝)
        a✝ : Eq (f x✝) (f' x✝)
        ⊢ Eq ((fun x => { fst := f x, snd := g x }) x✝) ((fun x => { fst := f' x, snd  …
      -/
      simp only [*]
      /-
        🎉 no goals
      -/

-- See `EventuallyEq.comp_tendsto` further below for a similar statement w.r.t.
-- composition on the right.

theorem EventuallyEq.fun_comp {f g : α → β} {l : Filter α} (H : f =ᶠ[l] g) (h : β → γ) :
    h ∘ f =ᶠ[l] h ∘ g :=
  H.mono fun _ hx => congr_arg h hx


theorem EventuallyEq.comp₂ {δ} {f f' : α → β} {g g' : α → γ} {l} (Hf : f =ᶠ[l] f') (h : β → γ → δ)
    (Hg : g =ᶠ[l] g') : (fun x => h (f x) (g x)) =ᶠ[l] fun x => h (f' x) (g' x) :=
  (Hf.prod_mk Hg).fun_comp (uncurry h)


@[to_additive]
theorem EventuallyEq.mul [Mul β] {f f' g g' : α → β} {l : Filter α} (h : f =ᶠ[l] g)
    (h' : f' =ᶠ[l] g') : (fun x => f x * f' x) =ᶠ[l] fun x => g x * g' x :=
  h.comp₂ (· * ·) h'


@[to_additive const_smul]
theorem EventuallyEq.pow_const {γ} [Pow β γ] {f g : α → β} {l : Filter α} (h : f =ᶠ[l] g) (c : γ) :
    (fun x => f x ^ c) =ᶠ[l] fun x => g x ^ c :=
  h.fun_comp (· ^ c)


@[to_additive]
theorem EventuallyEq.inv [Inv β] {f g : α → β} {l : Filter α} (h : f =ᶠ[l] g) :
    (fun x => (f x)⁻¹) =ᶠ[l] fun x => (g x)⁻¹ :=
  h.fun_comp Inv.inv


@[to_additive]
theorem EventuallyEq.div [Div β] {f f' g g' : α → β} {l : Filter α} (h : f =ᶠ[l] g)
    (h' : f' =ᶠ[l] g') : (fun x => f x / f' x) =ᶠ[l] fun x => g x / g' x :=
  h.comp₂ (· / ·) h'


attribute [to_additive] EventuallyEq.const_smul


@[to_additive]
theorem EventuallyEq.smul {𝕜} [SMul 𝕜 β] {l : Filter α} {f f' : α → 𝕜} {g g' : α → β}
    (hf : f =ᶠ[l] f') (hg : g =ᶠ[l] g') : (fun x => f x • g x) =ᶠ[l] fun x => f' x • g' x :=
  hf.comp₂ (· • ·) hg


theorem EventuallyEq.sup [Max β] {l : Filter α} {f f' g g' : α → β} (hf : f =ᶠ[l] f')
    (hg : g =ᶠ[l] g') : (fun x => f x ⊔ g x) =ᶠ[l] fun x => f' x ⊔ g' x :=
  hf.comp₂ (· ⊔ ·) hg


theorem EventuallyEq.inf [Min β] {l : Filter α} {f f' g g' : α → β} (hf : f =ᶠ[l] f')
    (hg : g =ᶠ[l] g') : (fun x => f x ⊓ g x) =ᶠ[l] fun x => f' x ⊓ g' x :=
  hf.comp₂ (· ⊓ ·) hg


theorem EventuallyEq.preimage {l : Filter α} {f g : α → β} (h : f =ᶠ[l] g) (s : Set β) :
    f ⁻¹' s =ᶠ[l] g ⁻¹' s :=
  h.fun_comp s


theorem EventuallyEq.inter {s t s' t' : Set α} {l : Filter α} (h : s =ᶠ[l] t) (h' : s' =ᶠ[l] t') :
    (s ∩ s' : Set α) =ᶠ[l] (t ∩ t' : Set α) :=
  h.comp₂ (· ∧ ·) h'


theorem EventuallyEq.union {s t s' t' : Set α} {l : Filter α} (h : s =ᶠ[l] t) (h' : s' =ᶠ[l] t') :
    (s ∪ s' : Set α) =ᶠ[l] (t ∪ t' : Set α) :=
  h.comp₂ (· ∨ ·) h'


theorem EventuallyEq.compl {s t : Set α} {l : Filter α} (h : s =ᶠ[l] t) :
    (sᶜ : Set α) =ᶠ[l] (tᶜ : Set α) :=
  h.fun_comp Not


theorem EventuallyEq.diff {s t s' t' : Set α} {l : Filter α} (h : s =ᶠ[l] t) (h' : s' =ᶠ[l] t') :
    (s \ s' : Set α) =ᶠ[l] (t \ t' : Set α) :=
  h.inter h'.compl


protected theorem EventuallyEq.symmDiff {s t s' t' : Set α} {l : Filter α}
    (h : s =ᶠ[l] t) (h' : s' =ᶠ[l] t') : (s ∆ s' : Set α) =ᶠ[l] (t ∆ t' : Set α) :=
  (h.diff h').union (h'.diff h)


theorem eventuallyEq_empty {s : Set α} {l : Filter α} : s =ᶠ[l] (∅ : Set α) ↔ ∀ᶠ x in l, x ∉ s :=
                               /-
                                 α : Type u
                                 s : Set α
                                 l : Filter α
                                 ⊢ Iff (Filter.Eventually (fun x => Iff (Membership.mem s x) (Membership.mem Em …
                               -/
  eventuallyEq_set.trans <| by simp
                               /-
                                 🎉 no goals
                               -/


theorem inter_eventuallyEq_left {s t : Set α} {l : Filter α} :
    (s ∩ t : Set α) =ᶠ[l] s ↔ ∀ᶠ x in l, x ∈ s → x ∈ t := by
  /-
    α : Type u
    s t : Set α
    l : Filter α
    ⊢ Iff (l.EventuallyEq (Inter.inter s t) s) (Filter.Eventually (fun x => Member …
  -/
  simp only [eventuallyEq_set, mem_inter_iff, and_iff_left_iff_imp]
  /-
    🎉 no goals
  -/


theorem inter_eventuallyEq_right {s t : Set α} {l : Filter α} :
    (s ∩ t : Set α) =ᶠ[l] t ↔ ∀ᶠ x in l, x ∈ t → x ∈ s := by
  /-
    α : Type u
    s t : Set α
    l : Filter α
    ⊢ Iff (l.EventuallyEq (Inter.inter s t) t) (Filter.Eventually (fun x => Member …
  -/
  rw [inter_comm, inter_eventuallyEq_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem eventuallyEq_principal {s : Set α} {f g : α → β} : f =ᶠ[𝓟 s] g ↔ EqOn f g s :=
  Iff.rfl


theorem eventuallyEq_inf_principal_iff {F : Filter α} {s : Set α} {f g : α → β} :
    f =ᶠ[F ⊓ 𝓟 s] g ↔ ∀ᶠ x in F, x ∈ s → f x = g x :=
  eventually_inf_principal


theorem EventuallyEq.sub_eq [AddGroup β] {f g : α → β} {l : Filter α} (h : f =ᶠ[l] g) :
                        /-
                          α : Type u
                          β : Type v
                          inst✝ : AddGroup β
                          f g : α → β
                          l : Filter α
                          h : l.EventuallyEq f g
                          ⊢ l.EventuallyEq (HSub.hSub f g) 0
                        -/
    f - g =ᶠ[l] 0 := by simpa using ((EventuallyEq.refl l f).sub h).symm
                        /-
                          🎉 no goals
                        -/


theorem eventuallyEq_iff_sub [AddGroup β] {f g : α → β} {l : Filter α} :
    f =ᶠ[l] g ↔ f - g =ᶠ[l] 0 :=
                                  /-
                                    α : Type u
                                    β : Type v
                                    inst✝ : AddGroup β
                                    f g : α → β
                                    l : Filter α
                                    h : l.EventuallyEq (HSub.hSub f g) 0
                                    ⊢ l.EventuallyEq f g
                                  -/
  ⟨fun h => h.sub_eq, fun h => by simpa using h.add (EventuallyEq.refl l g)⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem eventuallyEq_iff_all_subsets {f g : α → β} {l : Filter α} :
    f =ᶠ[l] g ↔ ∀ s : Set α, ∀ᶠ x in l, x ∈ s → f x = g x :=
  eventually_iff_all_subsets


theorem EventuallyLE.congr {f f' g g' : α → β} (H : f ≤ᶠ[l] g) (hf : f =ᶠ[l] f') (hg : g =ᶠ[l] g') :
    f' ≤ᶠ[l] g' :=
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 inst✝ : LE β
                                                 l : Filter α
                                                 f f' g g' : α → β
                                                 H✝ : l.EventuallyLE f g
                                                 hf✝ : l.EventuallyEq f f'
                                                 hg✝ : l.EventuallyEq g g'
                                                 x : α
                                                 hf : Eq (f x) (f' x)
                                                 hg : Eq (g x) (g' x)
                                                 H : LE.le (f x) (g x)
                                                 ⊢ LE.le (f' x) (g' x)
                                               -/
  H.mp <| hg.mp <| hf.mono fun x hf hg H => by rwa [hf, hg] at H
                                               /-
                                                 🎉 no goals
                                               -/


theorem eventuallyLE_congr {f f' g g' : α → β} (hf : f =ᶠ[l] f') (hg : g =ᶠ[l] g') :
    f ≤ᶠ[l] g ↔ f' ≤ᶠ[l] g' :=
  ⟨fun H => H.congr hf hg, fun H => H.congr hf.symm hg.symm⟩


theorem eventuallyLE_iff_all_subsets {f g : α → β} {l : Filter α} :
    f ≤ᶠ[l] g ↔ ∀ s : Set α, ∀ᶠ x in l, x ∈ s → f x ≤ g x :=
  eventually_iff_all_subsets


theorem EventuallyEq.le (h : f =ᶠ[l] g) : f ≤ᶠ[l] g :=
  h.mono fun _ => le_of_eq


@[refl]
theorem EventuallyLE.refl (l : Filter α) (f : α → β) : f ≤ᶠ[l] f :=
  EventuallyEq.rfl.le


theorem EventuallyLE.rfl : f ≤ᶠ[l] f :=
  EventuallyLE.refl l f


@[trans]
theorem EventuallyLE.trans (H₁ : f ≤ᶠ[l] g) (H₂ : g ≤ᶠ[l] h) : f ≤ᶠ[l] h :=
  H₂.mp <| H₁.mono fun _ => le_trans


instance : Trans ((· ≤ᶠ[l] ·) : (α → β) → (α → β) → Prop) (· ≤ᶠ[l] ·) (· ≤ᶠ[l] ·) where
  trans := EventuallyLE.trans


@[trans]
theorem EventuallyEq.trans_le (H₁ : f =ᶠ[l] g) (H₂ : g ≤ᶠ[l] h) : f ≤ᶠ[l] h :=
  H₁.le.trans H₂


instance : Trans ((· =ᶠ[l] ·) : (α → β) → (α → β) → Prop) (· ≤ᶠ[l] ·) (· ≤ᶠ[l] ·) where
  trans := EventuallyEq.trans_le


@[trans]
theorem EventuallyLE.trans_eq (H₁ : f ≤ᶠ[l] g) (H₂ : g =ᶠ[l] h) : f ≤ᶠ[l] h :=
  H₁.trans H₂.le


instance : Trans ((· ≤ᶠ[l] ·) : (α → β) → (α → β) → Prop) (· =ᶠ[l] ·) (· ≤ᶠ[l] ·) where
  trans := EventuallyLE.trans_eq


theorem EventuallyLE.antisymm [PartialOrder β] {l : Filter α} {f g : α → β} (h₁ : f ≤ᶠ[l] g)
    (h₂ : g ≤ᶠ[l] f) : f =ᶠ[l] g :=
  h₂.mp <| h₁.mono fun _ => le_antisymm


theorem eventuallyLE_antisymm_iff [PartialOrder β] {l : Filter α} {f g : α → β} :
    f =ᶠ[l] g ↔ f ≤ᶠ[l] g ∧ g ≤ᶠ[l] f := by
  /-
    α : Type u
    β : Type v
    inst✝ : PartialOrder β
    l : Filter α
    f g : α → β
    ⊢ Iff (l.EventuallyEq f g) (And (l.EventuallyLE f g) (l.EventuallyLE g f))
  -/
  simp only [EventuallyEq, EventuallyLE, le_antisymm_iff, eventually_and]
  /-
    🎉 no goals
  -/


theorem EventuallyLE.le_iff_eq [PartialOrder β] {l : Filter α} {f g : α → β} (h : f ≤ᶠ[l] g) :
    g ≤ᶠ[l] f ↔ g =ᶠ[l] f :=
  ⟨fun h' => h'.antisymm h, EventuallyEq.le⟩


theorem Eventually.ne_of_lt [Preorder β] {l : Filter α} {f g : α → β} (h : ∀ᶠ x in l, f x < g x) :
    ∀ᶠ x in l, f x ≠ g x :=
  h.mono fun _ hx => hx.ne


theorem Eventually.ne_top_of_lt [PartialOrder β] [OrderTop β] {l : Filter α} {f g : α → β}
    (h : ∀ᶠ x in l, f x < g x) : ∀ᶠ x in l, f x ≠ ⊤ :=
  h.mono fun _ hx => hx.ne_top


theorem Eventually.lt_top_of_ne [PartialOrder β] [OrderTop β] {l : Filter α} {f : α → β}
    (h : ∀ᶠ x in l, f x ≠ ⊤) : ∀ᶠ x in l, f x < ⊤ :=
  h.mono fun _ hx => hx.lt_top


theorem Eventually.lt_top_iff_ne_top [PartialOrder β] [OrderTop β] {l : Filter α} {f : α → β} :
    (∀ᶠ x in l, f x < ⊤) ↔ ∀ᶠ x in l, f x ≠ ⊤ :=
  ⟨Eventually.ne_of_lt, Eventually.lt_top_of_ne⟩


@[mono]
theorem EventuallyLE.inter {s t s' t' : Set α} {l : Filter α} (h : s ≤ᶠ[l] t) (h' : s' ≤ᶠ[l] t') :
    (s ∩ s' : Set α) ≤ᶠ[l] (t ∩ t' : Set α) :=
  h'.mp <| h.mono fun _ => And.imp


@[mono]
theorem EventuallyLE.union {s t s' t' : Set α} {l : Filter α} (h : s ≤ᶠ[l] t) (h' : s' ≤ᶠ[l] t') :
    (s ∪ s' : Set α) ≤ᶠ[l] (t ∪ t' : Set α) :=
  h'.mp <| h.mono fun _ => Or.imp


@[mono]
theorem EventuallyLE.compl {s t : Set α} {l : Filter α} (h : s ≤ᶠ[l] t) :
    (tᶜ : Set α) ≤ᶠ[l] (sᶜ : Set α) :=
  h.mono fun _ => mt


@[mono]
theorem EventuallyLE.diff {s t s' t' : Set α} {l : Filter α} (h : s ≤ᶠ[l] t) (h' : t' ≤ᶠ[l] s') :
    (s \ s' : Set α) ≤ᶠ[l] (t \ t' : Set α) :=
  h.inter h'.compl


theorem set_eventuallyLE_iff_mem_inf_principal {s t : Set α} {l : Filter α} :
    s ≤ᶠ[l] t ↔ t ∈ l ⊓ 𝓟 s :=
  eventually_inf_principal.symm


theorem set_eventuallyLE_iff_inf_principal_le {s t : Set α} {l : Filter α} :
    s ≤ᶠ[l] t ↔ l ⊓ 𝓟 s ≤ l ⊓ 𝓟 t :=
  set_eventuallyLE_iff_mem_inf_principal.trans <| by
    /-
      α : Type u
      s t : Set α
      l : Filter α
      ⊢ Iff (Membership.mem (Min.min l (Filter.principal s)) t) (LE.le (Min.min l (F …
    -/
    simp only [le_inf_iff, inf_le_left, true_and, le_principal_iff]
    /-
      🎉 no goals
    -/


theorem set_eventuallyEq_iff_inf_principal {s t : Set α} {l : Filter α} :
    s =ᶠ[l] t ↔ l ⊓ 𝓟 s = l ⊓ 𝓟 t := by
  /-
    α : Type u
    s t : Set α
    l : Filter α
    ⊢ Iff (l.EventuallyEq s t) (Eq (Min.min l (Filter.principal s)) (Min.min l (Fi …
  -/
  simp only [eventuallyLE_antisymm_iff, le_antisymm_iff, set_eventuallyLE_iff_inf_principal_le]
  /-
    🎉 no goals
  -/


theorem EventuallyLE.sup [SemilatticeSup β] {l : Filter α} {f₁ f₂ g₁ g₂ : α → β} (hf : f₁ ≤ᶠ[l] f₂)
    (hg : g₁ ≤ᶠ[l] g₂) : f₁ ⊔ g₁ ≤ᶠ[l] f₂ ⊔ g₂ := by
  /-
    α : Type u
    β : Type v
    inst✝ : SemilatticeSup β
    l : Filter α
    f₁ f₂ g₁ g₂ : α → β
    hf : l.EventuallyLE f₁ f₂
    hg : l.EventuallyLE g₁ g₂
    ⊢ l.EventuallyLE (Max.max f₁ g₁) (Max.max f₂ g₂)
  -/
  filter_upwards [hf, hg] with x hfx hgx using sup_le_sup hfx hgx
  /-
    🎉 no goals
  -/


theorem EventuallyLE.sup_le [SemilatticeSup β] {l : Filter α} {f g h : α → β} (hf : f ≤ᶠ[l] h)
    (hg : g ≤ᶠ[l] h) : f ⊔ g ≤ᶠ[l] h := by
  /-
    α : Type u
    β : Type v
    inst✝ : SemilatticeSup β
    l : Filter α
    f g h : α → β
    hf : l.EventuallyLE f h
    hg : l.EventuallyLE g h
    ⊢ l.EventuallyLE (Max.max f g) h
  -/
  filter_upwards [hf, hg] with x hfx hgx using _root_.sup_le hfx hgx
  /-
    🎉 no goals
  -/


theorem EventuallyLE.le_sup_of_le_left [SemilatticeSup β] {l : Filter α} {f g h : α → β}
    (hf : h ≤ᶠ[l] f) : h ≤ᶠ[l] f ⊔ g :=
  hf.mono fun _ => _root_.le_sup_of_le_left


theorem EventuallyLE.le_sup_of_le_right [SemilatticeSup β] {l : Filter α} {f g h : α → β}
    (hg : h ≤ᶠ[l] g) : h ≤ᶠ[l] f ⊔ g :=
  hg.mono fun _ => _root_.le_sup_of_le_right


theorem join_le {f : Filter (Filter α)} {l : Filter α} (h : ∀ᶠ m in f, m ≤ l) : join f ≤ l :=
  fun _ hs => h.mono fun _ hm => hm hs


@[simp]
theorem map_principal {s : Set α} {f : α → β} : map f (𝓟 s) = 𝓟 (Set.image f s) :=
  Filter.ext fun _ => image_subset_iff.symm


@[simp]
theorem eventually_map {P : β → Prop} : (∀ᶠ b in map m f, P b) ↔ ∀ᶠ a in f, P (m a) :=
  Iff.rfl


@[simp]
theorem frequently_map {P : β → Prop} : (∃ᶠ b in map m f, P b) ↔ ∃ᶠ a in f, P (m a) :=
  Iff.rfl


@[simp]
theorem mem_map : t ∈ map m f ↔ m ⁻¹' t ∈ f :=
  Iff.rfl


theorem mem_map' : t ∈ map m f ↔ { x | m x ∈ t } ∈ f :=
  Iff.rfl


theorem image_mem_map (hs : s ∈ f) : m '' s ∈ map m f :=
  f.sets_of_superset hs <| subset_preimage_image m s

-- The simpNF linter says that the LHS can be simplified via `Filter.mem_map`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[simp 1100, nolint simpNF]
theorem image_mem_map_iff (hf : Injective m) : m '' s ∈ map m f ↔ s ∈ f :=
               /-
                 α : Type u
                 β : Type v
                 f : Filter α
                 m : α → β
                 s : Set α
                 hf : Function.Injective m
                 h : Membership.mem (Filter.map m f) (Set.image m s)
                 ⊢ Membership.mem f s
               -/
  ⟨fun h => by rwa [← preimage_image_eq s hf], image_mem_map⟩
               /-
                 🎉 no goals
               -/


theorem range_mem_map : range m ∈ map m f := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    m : α → β
    ⊢ Membership.mem (Filter.map m f) (Set.range m)
  -/
  rw [← image_univ]
  /-
    α : Type u
    β : Type v
    f : Filter α
    m : α → β
    ⊢ Membership.mem (Filter.map m f) (Set.image m Set.univ)
  -/
  exact image_mem_map univ_mem
  /-
    🎉 no goals
  -/


theorem mem_map_iff_exists_image : t ∈ map m f ↔ ∃ s ∈ f, m '' s ⊆ t :=
  ⟨fun ht => ⟨m ⁻¹' t, ht, image_preimage_subset _ _⟩, fun ⟨_, hs, ht⟩ =>
    mem_of_superset (image_mem_map hs) ht⟩


@[simp]
theorem map_id : Filter.map id f = f :=
  filter_eq <| rfl


@[simp]
theorem map_id' : Filter.map (fun x => x) f = f :=
  map_id


@[simp]
theorem map_compose : Filter.map m' ∘ Filter.map m = Filter.map (m' ∘ m) :=
  funext fun _ => filter_eq <| rfl


@[simp]
theorem map_map : Filter.map m' (Filter.map m f) = Filter.map (m' ∘ m) f :=
  congr_fun Filter.map_compose f


/-- If functions `m₁` and `m₂` are eventually equal at a filter `f`, then
they map this filter to the same filter. -/
theorem map_congr {m₁ m₂ : α → β} {f : Filter α} (h : m₁ =ᶠ[f] m₂) : map m₁ f = map m₂ f :=
  Filter.ext' fun _ => eventually_congr (h.mono fun _ hx => hx ▸ Iff.rfl)


theorem mem_comap' : s ∈ comap f l ↔ { y | ∀ ⦃x⦄, f x = y → x ∈ s } ∈ l :=
                                                                                       /-
                                                                                         α : Type u
                                                                                         β : Type v
                                                                                         f : α → β
                                                                                         l : Filter β
                                                                                         s : Set α
                                                                                         x✝ : Membership.mem (Filter.comap f l) s
                                                                                         t : Set β
                                                                                         ht : Membership.mem l t
                                                                                         hts : HasSubset.Subset (Set.preimage f t) s
                                                                                         y : β
                                                                                         hy : Membership.mem t y
                                                                                         x : α
                                                                                         hx : Eq (f x) y
                                                                                         ⊢ Membership.mem t (f x)
                                                                                       -/
  ⟨fun ⟨t, ht, hts⟩ => mem_of_superset ht fun y hy x hx => hts <| mem_preimage.2 <| by rwa [hx],
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    fun h => ⟨_, h, fun _ hx => hx rfl⟩⟩

-- TODO: it would be nice to use `kernImage` much more to take advantage of common name and API,
-- and then this would become `mem_comap'`

theorem mem_comap'' : s ∈ comap f l ↔ kernImage f s ∈ l :=
  mem_comap'


/-- RHS form is used, e.g., in the definition of `UniformSpace`. -/
lemma mem_comap_prod_mk {x : α} {s : Set β} {F : Filter (α × β)} :
    s ∈ comap (Prod.mk x) F ↔ {p : α × β | p.fst = x → p.snd ∈ s} ∈ F := by
  /-
    α : Type u
    β : Type v
    x : α
    s : Set β
    F : Filter (Prod α β)
    ⊢ Iff (Membership.mem (Filter.comap (Prod.mk x) F) s) (Membership.mem F (setOf …
  -/
  simp_rw [mem_comap', Prod.ext_iff, and_imp, @forall_swap β (_ = _), forall_eq, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem eventually_comap : (∀ᶠ a in comap f l, p a) ↔ ∀ᶠ b in l, ∀ a, f a = b → p a :=
  mem_comap'


@[simp]
theorem frequently_comap : (∃ᶠ a in comap f l, p a) ↔ ∃ᶠ b in l, ∃ a, f a = b ∧ p a := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l : Filter β
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun a => p a) (Filter.comap f l)) (Filter.Frequently …
  -/
  simp only [Filter.Frequently, eventually_comap, not_exists, _root_.not_and]
  /-
    🎉 no goals
  -/


theorem mem_comap_iff_compl : s ∈ comap f l ↔ (f '' sᶜ)ᶜ ∈ l := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l : Filter β
    s : Set α
    ⊢ Iff (Membership.mem (Filter.comap f l) s) (Membership.mem l (HasCompl.compl  …
  -/
  simp only [mem_comap'', kernImage_eq_compl]
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 α : Type u
                                                                 β : Type v
                                                                 f : α → β
                                                                 l : Filter β
                                                                 s : Set α
                                                                 ⊢ Iff (Membership.mem (Filter.comap f l) (HasCompl.compl s)) (Membership.mem l …
                                                               -/
theorem compl_mem_comap : sᶜ ∈ comap f l ↔ (f '' s)ᶜ ∈ l := by rw [mem_comap_iff_compl, compl_compl]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The analog of `kernImage` for filters. A set `s` belongs to `Filter.kernMap m f` if either of
the following equivalent conditions hold.

1. There exists a set `t ∈ f` such that `s = kernImage m t`. This is used as a definition.
2. There exists a set `t` such that `tᶜ ∈ f` and `sᶜ = m '' t`, see `Filter.mem_kernMap_iff_compl`
and `Filter.compl_mem_kernMap`.

This definition because it gives a right adjoint to `Filter.comap`, and because it has a nice
interpretation when working with `co-` filters (`Filter.cocompact`, `Filter.cofinite`, ...).
For example, `kernMap m (cocompact α)` is the filter generated by the complements of the sets
`m '' K` where `K` is a compact subset of `α`. -/
def kernMap (m : α → β) (f : Filter α) : Filter β where
  sets := (kernImage m) '' f.sets
                                      /-
                                        α : Type u
                                        β : Type v
                                        γ : Type w
                                        δ : Type u_1
                                        ι : Sort x
                                        m : α → β
                                        f : Filter α
                                        ⊢ Eq (Set.kernImage m Set.univ) Set.univ
                                      -/
  univ_sets := ⟨univ, f.univ_sets, by simp [kernImage_eq_compl]⟩
                                      /-
                                        🎉 no goals
                                      -/
  sets_of_superset := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type u_1
      ι : Sort x
      m : α → β
      f : Filter α
      ⊢ ∀ {x y : Set β}, Membership.mem (Set.image (Set.kernImage m) f.sets) x → Has …
    -/
    rintro _ t ⟨s, hs, rfl⟩ hst
    /-
      case intro.intro
      α : Type u
      β : Type v
      γ : Type w
      δ : Type u_1
      ι : Sort x
      m : α → β
      f : Filter α
      t : Set β
      s : Set α
      hs : Membership.mem f.sets s
      hst : HasSubset.Subset (Set.kernImage m s) t
      ⊢ Membership.mem (Set.image (Set.kernImage m) f.sets) t
    -/
    refine ⟨s ∪ m ⁻¹' t, mem_of_superset hs subset_union_left, ?_⟩
    /-
      case intro.intro
      α : Type u
      β : Type v
      γ : Type w
      δ : Type u_1
      ι : Sort x
      m : α → β
      f : Filter α
      t : Set β
      s : Set α
      hs : Membership.mem f.sets s
      hst : HasSubset.Subset (Set.kernImage m s) t
      ⊢ Eq (Set.kernImage m (Union.union s (Set.preimage m t))) t
    -/
    rw [kernImage_union_preimage, union_eq_right.mpr hst]
    /-
      🎉 no goals
    -/
  inter_sets := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      δ : Type u_1
      ι : Sort x
      m : α → β
      f : Filter α
      ⊢ ∀ {x y : Set β}, Membership.mem (Set.image (Set.kernImage m) f.sets) x → Mem …
    -/
    rintro _ _ ⟨s₁, h₁, rfl⟩ ⟨s₂, h₂, rfl⟩
    /-
      case intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      δ : Type u_1
      ι : Sort x
      m : α → β
      f : Filter α
      s₁ : Set α
      h₁ : Membership.mem f.sets s₁
      s₂ : Set α
      h₂ : Membership.mem f.sets s₂
      ⊢ Membership.mem (Set.image (Set.kernImage m) f.sets) (Inter.inter (Set.kernIm …
    -/
    exact ⟨s₁ ∩ s₂, f.inter_sets h₁ h₂, Set.preimage_kernImage.u_inf⟩
    /-
      🎉 no goals
    -/


theorem mem_kernMap {s : Set β} : s ∈ kernMap m f ↔ ∃ t ∈ f, kernImage m t = s :=
  Iff.rfl


theorem mem_kernMap_iff_compl {s : Set β} : s ∈ kernMap m f ↔ ∃ t, tᶜ ∈ f ∧ m '' t = sᶜ := by
  /-
    α : Type u
    β : Type v
    m : α → β
    f : Filter α
    s : Set β
    ⊢ Iff (Membership.mem (Filter.kernMap m f) s) (Exists fun t => And (Membership …
  -/
  rw [mem_kernMap, compl_surjective.exists]
  /-
    α : Type u
    β : Type v
    m : α → β
    f : Filter α
    s : Set β
    ⊢ Iff (Exists fun x => And (Membership.mem f (HasCompl.compl x)) (Eq (Set.kern …
  -/
  refine exists_congr (fun x ↦ and_congr_right fun _ ↦ ?_)
  /-
    α : Type u
    β : Type v
    m : α → β
    f : Filter α
    s : Set β
    x : Set α
    x✝ : Membership.mem f (HasCompl.compl x)
    ⊢ Iff (Eq (Set.kernImage m (HasCompl.compl x)) s) (Eq (Set.image m x) (HasComp …
  -/
  rw [kernImage_compl, compl_eq_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem compl_mem_kernMap {s : Set β} : sᶜ ∈ kernMap m f ↔ ∃ t, tᶜ ∈ f ∧ m '' t = s := by
  /-
    α : Type u
    β : Type v
    m : α → β
    f : Filter α
    s : Set β
    ⊢ Iff (Membership.mem (Filter.kernMap m f) (HasCompl.compl s)) (Exists fun t = …
  -/
  simp_rw [mem_kernMap_iff_compl, compl_compl]
  /-
    🎉 no goals
  -/


instance : LawfulFunctor (Filter : Type u → Type u) where
  id_map _ := map_id
  comp_map _ _ _ := map_map.symm
  map_const := rfl


theorem pure_sets (a : α) : (pure a : Filter α).sets = { s | a ∈ s } :=
  rfl


@[simp]
theorem eventually_pure {a : α} {p : α → Prop} : (∀ᶠ x in pure a, p x) ↔ p a :=
  Iff.rfl


@[simp]
theorem principal_singleton (a : α) : 𝓟 {a} = pure a :=
                         /-
                           α : Type u
                           a : α
                           s : Set α
                           ⊢ Iff (Membership.mem (Filter.principal (Singleton.singleton a)) s) (Membershi …
                         -/
  Filter.ext fun s => by simp only [mem_pure, mem_principal, singleton_subset_iff]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem map_pure (f : α → β) (a : α) : map f (pure a) = pure (f a) :=
  rfl


theorem pure_le_principal {s : Set α} (a : α) : pure a ≤ 𝓟 s ↔ a ∈ s := by
  /-
    α : Type u
    s : Set α
    a : α
    ⊢ Iff (LE.le (Pure.pure a) (Filter.principal s)) (Membership.mem s a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] theorem join_pure (f : Filter α) : join (pure f) = f := rfl


@[simp]
theorem pure_bind (a : α) (m : α → Filter β) : bind (pure a) m = m a := by
  /-
    α : Type u
    β : Type v
    a : α
    m : α → Filter β
    ⊢ Eq ((Pure.pure a).bind m) (m a)
  -/
  simp only [Bind.bind, bind, map_pure, join_pure]
  /-
    🎉 no goals
  -/


theorem map_bind {α β} (m : β → γ) (f : Filter α) (g : α → Filter β) :
    map m (bind f g) = bind f (map m ∘ g) :=
  rfl


theorem bind_map {α β} (m : α → β) (f : Filter α) (g : β → Filter γ) :
    (bind (map m f) g) = bind f (g ∘ m) :=
  rfl


/-- The monad structure on filters. -/
protected def monad : Monad Filter where map := @Filter.map


protected theorem lawfulMonad : LawfulMonad Filter where
  map_const := rfl
  id_map _ := rfl
  seqLeft_eq _ _ := rfl
  seqRight_eq _ _ := rfl
  pure_seq _ _ := rfl
  bind_pure_comp _ _ := rfl
  bind_map _ _ := rfl
  pure_bind _ _ := rfl
  bind_assoc _ _ _ := rfl


instance : Alternative Filter where
  seq := fun x y => x.seq (y ())
  failure := ⊥
  orElse x y := x ⊔ y ()


@[simp]
theorem map_def {α β} (m : α → β) (f : Filter α) : m <$> f = map m f :=
  rfl


@[simp]
theorem bind_def {α β} (f : Filter α) (m : α → Filter β) : f >>= m = bind f m :=
  rfl


@[simp] theorem mem_comap : s ∈ comap m g ↔ ∃ t ∈ g, m ⁻¹' t ⊆ s := Iff.rfl


theorem preimage_mem_comap (ht : t ∈ g) : m ⁻¹' t ∈ comap m g :=
  ⟨t, ht, Subset.rfl⟩


theorem Eventually.comap {p : β → Prop} (hf : ∀ᶠ b in g, p b) (f : α → β) :
    ∀ᶠ a in comap f g, p (f a) :=
  preimage_mem_comap hf


theorem comap_id : comap id f = f :=
  le_antisymm (fun _ => preimage_mem_comap) fun _ ⟨_, ht, hst⟩ => mem_of_superset ht hst


theorem comap_id' : comap (fun x => x) f = f := comap_id


theorem comap_const_of_not_mem {x : β} (ht : t ∈ g) (hx : x ∉ t) : comap (fun _ : α => x) g = ⊥ :=
  empty_mem_iff_bot.1 <| mem_comap'.2 <| mem_of_superset ht fun _ hx' _ h => hx <| h.symm ▸ hx'


theorem comap_const_of_mem {x : β} (h : ∀ t ∈ g, x ∈ t) : comap (fun _ : α => x) g = ⊤ :=
  top_unique fun _ hs => univ_mem' fun _ => h _ (mem_comap'.1 hs) rfl


theorem map_const [NeBot f] {c : β} : (f.map fun _ => c) = pure c := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    inst✝ : f.NeBot
    c : β
    ⊢ Eq (Filter.map (fun x => c) f) (Pure.pure c)
  -/
  ext s
  /-
    case h
    α : Type u
    β : Type v
    f : Filter α
    inst✝ : f.NeBot
    c : β
    s : Set β
    ⊢ Iff (Membership.mem (Filter.map (fun x => c) f) s) (Membership.mem (Pure.pur …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : c ∈ s <;> simp [h]
                         /-
                           🎉 no goals
                         -/


theorem comap_comap {m : γ → β} {n : β → α} : comap m (comap n f) = comap (n ∘ m) f :=
                           /-
                             α : Type u
                             β : Type v
                             γ : Type w
                             f : Filter α
                             m : γ → β
                             n : β → α
                             s : Set γ
                             ⊢ Iff (Membership.mem (Filter.comap m (Filter.comap n f)) (HasCompl.compl s))  …
                           -/
  Filter.coext fun s => by simp only [compl_mem_comap, image_image, (· ∘ ·)]
                           /-
                             🎉 no goals
                           -/


theorem map_comm (H : ψ ∘ φ = ρ ∘ θ) (F : Filter α) :
    map ψ (map φ F) = map ρ (map θ F) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_1
    φ : α → β
    θ : α → γ
    ψ : β → δ
    ρ : γ → δ
    H : Eq (Function.comp ψ φ) (Function.comp ρ θ)
    F : Filter α
    ⊢ Eq (Filter.map ψ (Filter.map φ F)) (Filter.map ρ (Filter.map θ F))
  -/
  rw [Filter.map_map, H, ← Filter.map_map]
  /-
    🎉 no goals
  -/


theorem comap_comm (H : ψ ∘ φ = ρ ∘ θ) (G : Filter δ) :
    comap φ (comap ψ G) = comap θ (comap ρ G) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_1
    φ : α → β
    θ : α → γ
    ψ : β → δ
    ρ : γ → δ
    H : Eq (Function.comp ψ φ) (Function.comp ρ θ)
    G : Filter δ
    ⊢ Eq (Filter.comap φ (Filter.comap ψ G)) (Filter.comap θ (Filter.comap ρ G))
  -/
  rw [Filter.comap_comap, H, ← Filter.comap_comap]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Semiconj.filter_map {f : α → β} {ga : α → α} {gb : β → β}
    (h : Function.Semiconj f ga gb) : Function.Semiconj (map f) (map ga) (map gb) :=
  map_comm h.comp_eq


theorem _root_.Function.Commute.filter_map {f g : α → α} (h : Function.Commute f g) :
    Function.Commute (map f) (map g) :=
  h.semiconj.filter_map


theorem _root_.Function.Semiconj.filter_comap {f : α → β} {ga : α → α} {gb : β → β}
    (h : Function.Semiconj f ga gb) : Function.Semiconj (comap f) (comap gb) (comap ga) :=
  comap_comm h.comp_eq.symm


theorem _root_.Function.Commute.filter_comap {f g : α → α} (h : Function.Commute f g) :
    Function.Commute (comap f) (comap g) :=
  h.semiconj.filter_comap


theorem _root_.Function.LeftInverse.filter_map {f : α → β} {g : β → α} (hfg : LeftInverse g f) :
    LeftInverse (map g) (map f) := fun F ↦ by
  /-
    α : Type u
    β : Type v
    f : α → β
    g : β → α
    hfg : Function.LeftInverse g f
    F : Filter α
    ⊢ Eq (Filter.map g (Filter.map f F)) F
  -/
  rw [map_map, hfg.comp_eq_id, map_id]
  /-
    🎉 no goals
  -/


theorem _root_.Function.LeftInverse.filter_comap {f : α → β} {g : β → α} (hfg : LeftInverse g f) :
    RightInverse (comap g) (comap f) := fun F ↦ by
  /-
    α : Type u
    β : Type v
    f : α → β
    g : β → α
    hfg : Function.LeftInverse g f
    F : Filter α
    ⊢ Eq (Filter.comap f (Filter.comap g F)) F
  -/
  rw [comap_comap, hfg.comp_eq_id, comap_id]
  /-
    🎉 no goals
  -/


nonrec theorem _root_.Function.RightInverse.filter_map {f : α → β} {g : β → α}
    (hfg : RightInverse g f) : RightInverse (map g) (map f) :=
  hfg.filter_map


nonrec theorem _root_.Function.RightInverse.filter_comap {f : α → β} {g : β → α}
    (hfg : RightInverse g f) : LeftInverse (comap g) (comap f) :=
  hfg.filter_comap


theorem _root_.Set.LeftInvOn.filter_map_Iic {f : α → β} {g : β → α} (hfg : LeftInvOn g f s) :
    LeftInvOn (map g) (map f) (Iic <| 𝓟 s) := fun F (hF : F ≤ 𝓟 s) ↦ by
  /-
    α : Type u
    β : Type v
    s : Set α
    f : α → β
    g : β → α
    hfg : Set.LeftInvOn g f s
    F : Filter α
    hF : LE.le F (Filter.principal s)
    ⊢ Eq (Filter.map g (Filter.map f F)) F
  -/
  have : (g ∘ f) =ᶠ[𝓟 s] id := by simpa only [eventuallyEq_principal] using hfg
  /-
    α : Type u
    β : Type v
    s : Set α
    f : α → β
    g : β → α
    hfg : Set.LeftInvOn g f s
    F : Filter α
    hF : LE.le F (Filter.principal s)
    this : (Filter.principal s).EventuallyEq (Function.comp g f) id
    ⊢ Eq (Filter.map g (Filter.map f F)) F
  -/
  rw [map_map, map_congr (this.filter_mono hF), map_id]
  /-
    🎉 no goals
  -/


nonrec theorem _root_.Set.RightInvOn.filter_map_Iic {f : α → β} {g : β → α}
    (hfg : RightInvOn g f t) : RightInvOn (map g) (map f) (Iic <| 𝓟 t) :=
  hfg.filter_map_Iic


@[simp]
theorem comap_principal {t : Set β} : comap m (𝓟 t) = 𝓟 (m ⁻¹' t) :=
  Filter.ext fun _ => ⟨fun ⟨_u, hu, b⟩ => (preimage_mono hu).trans b,
    fun h => ⟨t, Subset.rfl, h⟩⟩


theorem principal_subtype {α : Type*} (s : Set α) (t : Set s) :
    𝓟 t = comap (↑) (𝓟 (((↑) : s → α) '' t)) := by
  /-
    α : Type u_2
    s : Set α
    t : Set ↑s
    ⊢ Eq (Filter.principal t) (Filter.comap Subtype.val (Filter.principal (Set.ima …
  -/
  rw [comap_principal, preimage_image_eq _ Subtype.coe_injective]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_pure {b : β} : comap m (pure b) = 𝓟 (m ⁻¹' {b}) := by
  /-
    α : Type u
    β : Type v
    m : α → β
    b : β
    ⊢ Eq (Filter.comap m (Pure.pure b)) (Filter.principal (Set.preimage m (Singlet …
  -/
  rw [← principal_singleton, comap_principal]
  /-
    🎉 no goals
  -/


theorem map_le_iff_le_comap : map m f ≤ g ↔ f ≤ comap m g :=
  ⟨fun h _ ⟨_, ht, hts⟩ => mem_of_superset (h ht) hts, fun h _ ht => h ⟨_, ht, Subset.rfl⟩⟩


theorem gc_map_comap (m : α → β) : GaloisConnection (map m) (comap m) :=
  fun _ _ => map_le_iff_le_comap


theorem comap_le_iff_le_kernMap : comap m g ≤ f ↔ g ≤ kernMap m f := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    g : Filter β
    m : α → β
    ⊢ Iff (LE.le (Filter.comap m g) f) (LE.le g (Filter.kernMap m f))
  -/
  simp [Filter.le_def, mem_comap'', mem_kernMap, -mem_comap]
  /-
    🎉 no goals
  -/


theorem gc_comap_kernMap (m : α → β) : GaloisConnection (comap m) (kernMap m) :=
  fun _ _ ↦ comap_le_iff_le_kernMap


theorem kernMap_principal {s : Set α} : kernMap m (𝓟 s) = 𝓟 (kernImage m s) := by
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    ⊢ Eq (Filter.kernMap m (Filter.principal s)) (Filter.principal (Set.kernImage  …
  -/
  refine eq_of_forall_le_iff (fun g ↦ ?_)
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Filter β
    ⊢ Iff (LE.le g (Filter.kernMap m (Filter.principal s))) (LE.le g (Filter.princ …
  -/
  rw [← comap_le_iff_le_kernMap, le_principal_iff, le_principal_iff, mem_comap'']
  /-
    🎉 no goals
  -/


@[mono]
theorem map_mono : Monotone (map m) :=
  (gc_map_comap m).monotone_l


@[mono]
theorem comap_mono : Monotone (comap m) :=
  (gc_map_comap m).monotone_u


/-- Temporary lemma that we can tag with `gcongr` -/
@[gcongr] theorem _root_.GCongr.Filter.map_le_map {F G : Filter α} (h : F ≤ G) :
    map m F ≤ map m G := map_mono h


/-- Temporary lemma that we can tag with `gcongr` -/
@[gcongr]
theorem _root_.GCongr.Filter.comap_le_comap {F G : Filter β} (h : F ≤ G) :
    comap m F ≤ comap m G := comap_mono h


@[simp] theorem map_bot : map m ⊥ = ⊥ := (gc_map_comap m).l_bot


@[simp] theorem map_sup : map m (f₁ ⊔ f₂) = map m f₁ ⊔ map m f₂ := (gc_map_comap m).l_sup


@[simp]
theorem map_iSup {f : ι → Filter α} : map m (⨆ i, f i) = ⨆ i, map m (f i) :=
  (gc_map_comap m).l_iSup


@[simp]
theorem map_top (f : α → β) : map f ⊤ = 𝓟 (range f) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Eq (Filter.map f Top.top) (Filter.principal (Set.range f))
  -/
  rw [← principal_univ, map_principal, image_univ]
  /-
    🎉 no goals
  -/


@[simp] theorem comap_top : comap m ⊤ = ⊤ := (gc_map_comap m).u_top


@[simp] theorem comap_inf : comap m (g₁ ⊓ g₂) = comap m g₁ ⊓ comap m g₂ := (gc_map_comap m).u_inf


@[simp]
theorem comap_iInf {f : ι → Filter β} : comap m (⨅ i, f i) = ⨅ i, comap m (f i) :=
  (gc_map_comap m).u_iInf


theorem le_comap_top (f : α → β) (l : Filter α) : l ≤ comap f ⊤ := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l : Filter α
    ⊢ LE.le l (Filter.comap f Top.top)
  -/
  rw [comap_top]
  /-
    α : Type u
    β : Type v
    f : α → β
    l : Filter α
    ⊢ LE.le l Top.top
  -/
  exact le_top
  /-
    🎉 no goals
  -/


theorem map_comap_le : map m (comap m g) ≤ g :=
  (gc_map_comap m).l_u_le _


theorem le_comap_map : f ≤ comap m (map m f) :=
  (gc_map_comap m).le_u_l _


@[simp]
theorem comap_bot : comap m ⊥ = ⊥ :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          m : α → β
                                          s : Set α
                                          x✝ : Membership.mem Bot.bot s
                                          ⊢ HasSubset.Subset (Set.preimage m EmptyCollection.emptyCollection) s
                                        -/
  bot_unique fun s _ => ⟨∅, mem_bot, by simp only [empty_subset, preimage_empty]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem neBot_of_comap (h : (comap m g).NeBot) : g.NeBot := by
  /-
    α : Type u
    β : Type v
    g : Filter β
    m : α → β
    h : (Filter.comap m g).NeBot
    ⊢ g.NeBot
  -/
  rw [neBot_iff] at *
  /-
    α : Type u
    β : Type v
    g : Filter β
    m : α → β
    h : Ne (Filter.comap m g) Bot.bot
    ⊢ Ne g Bot.bot
  -/
  contrapose! h
  /-
    α : Type u
    β : Type v
    g : Filter β
    m : α → β
    h : Eq g Bot.bot
    ⊢ Eq (Filter.comap m g) Bot.bot
  -/
  rw [h]
  /-
    α : Type u
    β : Type v
    g : Filter β
    m : α → β
    h : Eq g Bot.bot
    ⊢ Eq (Filter.comap m Bot.bot) Bot.bot
  -/
  exact comap_bot
  /-
    🎉 no goals
  -/


theorem comap_inf_principal_range : comap m (g ⊓ 𝓟 (range m)) = comap m g := by
  /-
    α : Type u
    β : Type v
    g : Filter β
    m : α → β
    ⊢ Eq (Filter.comap m (Min.min g (Filter.principal (Set.range m)))) (Filter.com …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem disjoint_comap (h : Disjoint g₁ g₂) : Disjoint (comap m g₁) (comap m g₂) := by
  /-
    α : Type u
    β : Type v
    g₁ g₂ : Filter β
    m : α → β
    h : Disjoint g₁ g₂
    ⊢ Disjoint (Filter.comap m g₁) (Filter.comap m g₂)
  -/
  simp only [disjoint_iff, ← comap_inf, h.eq_bot, comap_bot]
  /-
    🎉 no goals
  -/


theorem comap_iSup {ι} {f : ι → Filter β} {m : α → β} : comap m (iSup f) = ⨆ i, comap m (f i) :=
  (gc_comap_kernMap m).l_iSup


theorem comap_sSup {s : Set (Filter β)} {m : α → β} : comap m (sSup s) = ⨆ f ∈ s, comap m f := by
  /-
    α : Type u
    β : Type v
    s : Set (Filter β)
    m : α → β
    ⊢ Eq (Filter.comap m (SupSet.sSup s)) (iSup fun f => iSup fun h => Filter.coma …
  -/
  simp only [sSup_eq_iSup, comap_iSup, eq_self_iff_true]
  /-
    🎉 no goals
  -/


theorem comap_sup : comap m (g₁ ⊔ g₂) = comap m g₁ ⊔ comap m g₂ := by
  /-
    α : Type u
    β : Type v
    g₁ g₂ : Filter β
    m : α → β
    ⊢ Eq (Filter.comap m (Max.max g₁ g₂)) (Max.max (Filter.comap m g₁) (Filter.com …
  -/
  rw [sup_eq_iSup, comap_iSup, iSup_bool_eq, Bool.cond_true, Bool.cond_false]
  /-
    🎉 no goals
  -/


theorem map_comap (f : Filter β) (m : α → β) : (f.comap m).map m = f ⊓ 𝓟 (range m) := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    ⊢ Eq (Filter.map m (Filter.comap m f)) (Min.min f (Filter.principal (Set.range …
  -/
  refine le_antisymm (le_inf map_comap_le <| le_principal_iff.2 range_mem_map) ?_
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    ⊢ LE.le (Min.min f (Filter.principal (Set.range m))) (Filter.map m (Filter.com …
  -/
  rintro t' ⟨t, ht, sub⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    t' t : Set β
    ht : Membership.mem f t
    sub : HasSubset.Subset (Set.preimage m t) (Set.preimage m t')
    ⊢ Membership.mem (Min.min f (Filter.principal (Set.range m))) t'
  -/
  refine mem_inf_principal.2 (mem_of_superset ht ?_)
  /-
    case intro.intro
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    t' t : Set β
    ht : Membership.mem f t
    sub : HasSubset.Subset (Set.preimage m t) (Set.preimage m t')
    ⊢ HasSubset.Subset t (setOf fun x => Membership.mem (Set.range m) x → Membersh …
  -/
  rintro _ hxt ⟨x, rfl⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    t' t : Set β
    ht : Membership.mem f t
    sub : HasSubset.Subset (Set.preimage m t) (Set.preimage m t')
    x : α
    hxt : Membership.mem t (m x)
    ⊢ Membership.mem t' (m x)
  -/
  exact sub hxt
  /-
    🎉 no goals
  -/


theorem map_comap_setCoe_val (f : Filter β) (s : Set β) :
    (f.comap ((↑) : s → β)).map (↑) = f ⊓ 𝓟 s := by
  /-
    β : Type v
    f : Filter β
    s : Set β
    ⊢ Eq (Filter.map Subtype.val (Filter.comap Subtype.val f)) (Min.min f (Filter. …
  -/
  rw [map_comap, Subtype.range_val]
  /-
    🎉 no goals
  -/


theorem map_comap_of_mem {f : Filter β} {m : α → β} (hf : range m ∈ f) : (f.comap m).map m = f := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    hf : Membership.mem f (Set.range m)
    ⊢ Eq (Filter.map m (Filter.comap m f)) f
  -/
  rw [map_comap, inf_eq_left.2 (le_principal_iff.2 hf)]
  /-
    🎉 no goals
  -/


instance canLift (c) (p) [CanLift α β c p] :
    CanLift (Filter α) (Filter β) (map c) fun f => ∀ᶠ x : α in f, p x where
  prf f hf := ⟨comap c f, map_comap_of_mem <| hf.mono CanLift.prf⟩


theorem comap_le_comap_iff {f g : Filter β} {m : α → β} (hf : range m ∈ f) :
    comap m f ≤ comap m g ↔ f ≤ g :=
  ⟨fun h => map_comap_of_mem hf ▸ (map_mono h).trans map_comap_le, fun h => comap_mono h⟩


theorem map_comap_of_surjective {f : α → β} (hf : Surjective f) (l : Filter β) :
    map f (comap f l) = l :=
                         /-
                           α : Type u
                           β : Type v
                           f : α → β
                           hf : Function.Surjective f
                           l : Filter β
                           ⊢ Membership.mem l (Set.range f)
                         -/
  map_comap_of_mem <| by simp only [hf.range_eq, univ_mem]
                         /-
                           🎉 no goals
                         -/


theorem comap_injective {f : α → β} (hf : Surjective f) : Injective (comap f) :=
  LeftInverse.injective <| map_comap_of_surjective hf


theorem _root_.Function.Surjective.filter_map_top {f : α → β} (hf : Surjective f) : map f ⊤ = ⊤ :=
  (congr_arg _ comap_top).symm.trans <| map_comap_of_surjective hf ⊤


theorem subtype_coe_map_comap (s : Set α) (f : Filter α) :
                                                              /-
                                                                α : Type u
                                                                s : Set α
                                                                f : Filter α
                                                                ⊢ Eq (Filter.map Subtype.val (Filter.comap Subtype.val f)) (Min.min f (Filter. …
                                                              -/
    map ((↑) : s → α) (comap ((↑) : s → α) f) = f ⊓ 𝓟 s := by rw [map_comap, Subtype.range_coe]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem image_mem_of_mem_comap {f : Filter α} {c : β → α} (h : range c ∈ f) {W : Set β}
    (W_in : W ∈ comap c f) : c '' W ∈ f := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    c : β → α
    h : Membership.mem f (Set.range c)
    W : Set β
    W_in : Membership.mem (Filter.comap c f) W
    ⊢ Membership.mem f (Set.image c W)
  -/
  rw [← map_comap_of_mem h]
  /-
    α : Type u
    β : Type v
    f : Filter α
    c : β → α
    h : Membership.mem f (Set.range c)
    W : Set β
    W_in : Membership.mem (Filter.comap c f) W
    ⊢ Membership.mem (Filter.map c (Filter.comap c f)) (Set.image c W)
  -/
  exact image_mem_map W_in
  /-
    🎉 no goals
  -/


theorem image_coe_mem_of_mem_comap {f : Filter α} {U : Set α} (h : U ∈ f) {W : Set U}
    (W_in : W ∈ comap ((↑) : U → α) f) : (↑) '' W ∈ f :=
                             /-
                               α : Type u
                               f : Filter α
                               U : Set α
                               h : Membership.mem f U
                               W : Set ↑U
                               W_in : Membership.mem (Filter.comap Subtype.val f) W
                               ⊢ Membership.mem f (Set.range Subtype.val)
                             -/
  image_mem_of_mem_comap (by simp [h]) W_in
                             /-
                               🎉 no goals
                             -/


theorem comap_map {f : Filter α} {m : α → β} (h : Injective m) : comap m (map m f) = f :=
  le_antisymm
    (fun s hs =>
      mem_of_superset (preimage_mem_comap <| image_mem_map hs) <| by
        /-
          α : Type u
          β : Type v
          f : Filter α
          m : α → β
          h : Function.Injective m
          s : Set α
          hs : Membership.mem f s
          ⊢ HasSubset.Subset (Set.preimage m (Set.image m s)) s
        -/
        simp only [preimage_image_eq s h, Subset.rfl])
        /-
          🎉 no goals
        -/
    le_comap_map


theorem mem_comap_iff {f : Filter β} {m : α → β} (inj : Injective m) (large : Set.range m ∈ f)
    {S : Set α} : S ∈ comap m f ↔ m '' S ∈ f := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    inj : Function.Injective m
    large : Membership.mem f (Set.range m)
    S : Set α
    ⊢ Iff (Membership.mem (Filter.comap m f) S) (Membership.mem f (Set.image m S))
  -/
  rw [← image_mem_map_iff inj, map_comap_of_mem large]
  /-
    🎉 no goals
  -/


theorem map_le_map_iff_of_injOn {l₁ l₂ : Filter α} {f : α → β} {s : Set α} (h₁ : s ∈ l₁)
    (h₂ : s ∈ l₂) (hinj : InjOn f s) : map f l₁ ≤ map f l₂ ↔ l₁ ≤ l₂ :=
  ⟨fun h _t ht =>
    mp_mem h₁ <|
      mem_of_superset (h <| image_mem_map (inter_mem h₂ ht)) fun _y ⟨_x, ⟨hxs, hxt⟩, hxy⟩ hys =>
        hinj hxs hys hxy ▸ hxt,
    fun h => map_mono h⟩


theorem map_le_map_iff {f g : Filter α} {m : α → β} (hm : Injective m) :
                                    /-
                                      α : Type u
                                      β : Type v
                                      f g : Filter α
                                      m : α → β
                                      hm : Function.Injective m
                                      ⊢ Iff (LE.le (Filter.map m f) (Filter.map m g)) (LE.le f g)
                                    -/
    map m f ≤ map m g ↔ f ≤ g := by rw [map_le_iff_le_comap, comap_map hm]
                                    /-
                                      🎉 no goals
                                    -/


theorem map_eq_map_iff_of_injOn {f g : Filter α} {m : α → β} {s : Set α} (hsf : s ∈ f) (hsg : s ∈ g)
    (hm : InjOn m s) : map m f = map m g ↔ f = g := by
  simp only [le_antisymm_iff, map_le_map_iff_of_injOn hsf hsg hm,
    map_le_map_iff_of_injOn hsg hsf hm]


theorem map_inj {f g : Filter α} {m : α → β} (hm : Injective m) : map m f = map m g ↔ f = g :=
  map_eq_map_iff_of_injOn univ_mem univ_mem hm.injOn


theorem map_injective {m : α → β} (hm : Injective m) : Injective (map m) := fun _ _ =>
  (map_inj hm).1


theorem comap_neBot_iff {f : Filter β} {m : α → β} : NeBot (comap m f) ↔ ∀ t ∈ f, ∃ a, m a ∈ t := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    ⊢ Iff (Filter.comap m f).NeBot (∀ (t : Set β), Membership.mem f t → Exists fun …
  -/
  simp only [← forall_mem_nonempty_iff_neBot, mem_comap, forall_exists_index, and_imp]
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    ⊢ Iff (∀ (s : Set α) (x : Set β), Membership.mem f x → HasSubset.Subset (Set.p …
  -/
  exact ⟨fun h t t_in => h (m ⁻¹' t) t t_in Subset.rfl, fun h s t ht hst => (h t ht).imp hst⟩
  /-
    🎉 no goals
  -/


theorem comap_neBot {f : Filter β} {m : α → β} (hm : ∀ t ∈ f, ∃ a, m a ∈ t) : NeBot (comap m f) :=
  comap_neBot_iff.mpr hm


theorem comap_neBot_iff_frequently {f : Filter β} {m : α → β} :
    NeBot (comap m f) ↔ ∃ᶠ y in f, y ∈ range m := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    ⊢ Iff (Filter.comap m f).NeBot (Filter.Frequently (fun y => Membership.mem (Se …
  -/
  simp only [comap_neBot_iff, frequently_iff, mem_range, @and_comm (_ ∈ _), exists_exists_eq_and]
  /-
    🎉 no goals
  -/


theorem comap_neBot_iff_compl_range {f : Filter β} {m : α → β} :
    NeBot (comap m f) ↔ (range m)ᶜ ∉ f :=
  comap_neBot_iff_frequently


theorem comap_eq_bot_iff_compl_range {f : Filter β} {m : α → β} : comap m f = ⊥ ↔ (range m)ᶜ ∈ f :=
  not_iff_not.mp <| neBot_iff.symm.trans comap_neBot_iff_compl_range


theorem comap_surjective_eq_bot {f : Filter β} {m : α → β} (hm : Surjective m) :
    comap m f = ⊥ ↔ f = ⊥ := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    hm : Function.Surjective m
    ⊢ Iff (Eq (Filter.comap m f) Bot.bot) (Eq f Bot.bot)
  -/
  rw [comap_eq_bot_iff_compl_range, hm.range_eq, compl_univ, empty_mem_iff_bot]
  /-
    🎉 no goals
  -/


theorem disjoint_comap_iff (h : Surjective m) :
    Disjoint (comap m g₁) (comap m g₂) ↔ Disjoint g₁ g₂ := by
  /-
    α : Type u
    β : Type v
    g₁ g₂ : Filter β
    m : α → β
    h : Function.Surjective m
    ⊢ Iff (Disjoint (Filter.comap m g₁) (Filter.comap m g₂)) (Disjoint g₁ g₂)
  -/
  rw [disjoint_iff, disjoint_iff, ← comap_inf, comap_surjective_eq_bot h]
  /-
    🎉 no goals
  -/


theorem NeBot.comap_of_range_mem {f : Filter β} {m : α → β} (_ : NeBot f) (hm : range m ∈ f) :
    NeBot (comap m f) :=
  comap_neBot_iff_frequently.2 <| Eventually.frequently hm


@[simp]
theorem comap_fst_neBot_iff {f : Filter α} :
    (f.comap (Prod.fst : α × β → α)).NeBot ↔ f.NeBot ∧ Nonempty β := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    ⊢ Iff (Filter.comap Prod.fst f).NeBot (And f.NeBot (Nonempty β))
  -/
  cases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u
      β : Type v
      f : Filter α
      h✝ : IsEmpty β
      ⊢ Iff (Filter.comap Prod.fst f).NeBot (And f.NeBot (Nonempty β))
    -/
  · rw [filter_eq_bot_of_isEmpty (f.comap _), ← not_iff_not]; simp [*]
                                                              /-
                                                                🎉 no goals
                                                              -/
    /-
      case inr
      α : Type u
      β : Type v
      f : Filter α
      h✝ : Nonempty β
      ⊢ Iff (Filter.comap Prod.fst f).NeBot (And f.NeBot (Nonempty β))
    -/
  · simp [comap_neBot_iff_frequently, *]
    /-
      🎉 no goals
    -/


@[instance]
theorem comap_fst_neBot [Nonempty β] {f : Filter α} [NeBot f] :
    (f.comap (Prod.fst : α × β → α)).NeBot :=
  comap_fst_neBot_iff.2 ⟨‹_›, ‹_›⟩


@[simp]
theorem comap_snd_neBot_iff {f : Filter β} :
    (f.comap (Prod.snd : α × β → β)).NeBot ↔ Nonempty α ∧ f.NeBot := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    ⊢ Iff (Filter.comap Prod.snd f).NeBot (And (Nonempty α) f.NeBot)
  -/
  cases' isEmpty_or_nonempty α with hα hα
    /-
      case inl
      α : Type u
      β : Type v
      f : Filter β
      hα : IsEmpty α
      ⊢ Iff (Filter.comap Prod.snd f).NeBot (And (Nonempty α) f.NeBot)
    -/
  · rw [filter_eq_bot_of_isEmpty (f.comap _), ← not_iff_not]; simp
                                                              /-
                                                                🎉 no goals
                                                              -/
    /-
      case inr
      α : Type u
      β : Type v
      f : Filter β
      hα : Nonempty α
      ⊢ Iff (Filter.comap Prod.snd f).NeBot (And (Nonempty α) f.NeBot)
    -/
  · simp [comap_neBot_iff_frequently, hα]
    /-
      🎉 no goals
    -/


@[instance]
theorem comap_snd_neBot [Nonempty α] {f : Filter β} [NeBot f] :
    (f.comap (Prod.snd : α × β → β)).NeBot :=
  comap_snd_neBot_iff.2 ⟨‹_›, ‹_›⟩


theorem comap_eval_neBot_iff' {ι : Type*} {α : ι → Type*} {i : ι} {f : Filter (α i)} :
    (comap (eval i) f).NeBot ↔ (∀ j, Nonempty (α j)) ∧ NeBot f := by
  /-
    ι : Type u_2
    α : ι → Type u_3
    i : ι
    f : Filter (α i)
    ⊢ Iff (Filter.comap (Function.eval i) f).NeBot (And (∀ (j : ι), Nonempty (α j) …
  -/
  cases' isEmpty_or_nonempty (∀ j, α j) with H H
    /-
      case inl
      ι : Type u_2
      α : ι → Type u_3
      i : ι
      f : Filter (α i)
      H : IsEmpty ((j : ι) → α j)
      ⊢ Iff (Filter.comap (Function.eval i) f).NeBot (And (∀ (j : ι), Nonempty (α j) …
    -/
  · rw [filter_eq_bot_of_isEmpty (f.comap _), ← not_iff_not]
    /-
      case inl
      ι : Type u_2
      α : ι → Type u_3
      i : ι
      f : Filter (α i)
      H : IsEmpty ((j : ι) → α j)
      ⊢ Iff (Not Bot.bot.NeBot) (Not (And (∀ (j : ι), Nonempty (α j)) f.NeBot))
    -/
    simp [← Classical.nonempty_pi]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_2
      α : ι → Type u_3
      i : ι
      f : Filter (α i)
      H : Nonempty ((j : ι) → α j)
      ⊢ Iff (Filter.comap (Function.eval i) f).NeBot (And (∀ (j : ι), Nonempty (α j) …
    -/
  · have : ∀ j, Nonempty (α j) := Classical.nonempty_pi.1 H
    /-
      case inr
      ι : Type u_2
      α : ι → Type u_3
      i : ι
      f : Filter (α i)
      H : Nonempty ((j : ι) → α j)
      this : ∀ (j : ι), Nonempty (α j)
      ⊢ Iff (Filter.comap (Function.eval i) f).NeBot (And (∀ (j : ι), Nonempty (α j) …
    -/
    simp [comap_neBot_iff_frequently, *]
    /-
      🎉 no goals
    -/


@[simp]
theorem comap_eval_neBot_iff {ι : Type*} {α : ι → Type*} [∀ j, Nonempty (α j)] {i : ι}
                                                                  /-
                                                                    ι : Type u_2
                                                                    α : ι → Type u_3
                                                                    inst✝ : ∀ (j : ι), Nonempty (α j)
                                                                    i : ι
                                                                    f : Filter (α i)
                                                                    ⊢ Iff (Filter.comap (Function.eval i) f).NeBot f.NeBot
                                                                  -/
    {f : Filter (α i)} : (comap (eval i) f).NeBot ↔ NeBot f := by simp [comap_eval_neBot_iff', *]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[instance]
theorem comap_eval_neBot {ι : Type*} {α : ι → Type*} [∀ j, Nonempty (α j)] (i : ι)
    (f : Filter (α i)) [NeBot f] : (comap (eval i) f).NeBot :=
  comap_eval_neBot_iff.2 ‹_›


theorem comap_coe_neBot_of_le_principal {s : Set γ} {l : Filter γ} [h : NeBot l] (h' : l ≤ 𝓟 s) :
    NeBot (comap ((↑) : s → γ) l) :=
  h.comap_of_range_mem <| (@Subtype.range_coe γ s).symm ▸ h' (mem_principal_self s)


theorem NeBot.comap_of_surj {f : Filter β} {m : α → β} (hf : NeBot f) (hm : Surjective m) :
    NeBot (comap m f) :=
  hf.comap_of_range_mem <| univ_mem' hm


theorem NeBot.comap_of_image_mem {f : Filter β} {m : α → β} (hf : NeBot f) {s : Set α}
    (hs : m '' s ∈ f) : NeBot (comap m f) :=
  hf.comap_of_range_mem <| mem_of_superset hs (image_subset_range _ _)


@[simp]
theorem map_eq_bot_iff : map m f = ⊥ ↔ f = ⊥ :=
  ⟨by
    /-
      α : Type u
      β : Type v
      f : Filter α
      m : α → β
      ⊢ Eq (Filter.map m f) Bot.bot → Eq f Bot.bot
    -/
    rw [← empty_mem_iff_bot, ← empty_mem_iff_bot]
    /-
      α : Type u
      β : Type v
      f : Filter α
      m : α → β
      ⊢ Membership.mem (Filter.map m f) EmptyCollection.emptyCollection → Membership …
    -/
    /-
      🎉 no goals
    -/
    exact id, fun h => by simp only [h, map_bot]⟩
                          /-
                            🎉 no goals
                          -/


theorem map_neBot_iff (f : α → β) {F : Filter α} : NeBot (map f F) ↔ NeBot F := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    ⊢ Iff (Filter.map f F).NeBot F.NeBot
  -/
  simp only [neBot_iff, Ne, map_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem NeBot.map (hf : NeBot f) (m : α → β) : NeBot (map m f) :=
  (map_neBot_iff m).2 hf


theorem NeBot.of_map : NeBot (f.map m) → NeBot f :=
  (map_neBot_iff m).1


instance map_neBot [hf : NeBot f] : NeBot (f.map m) :=
  hf.map m


theorem sInter_comap_sets (f : α → β) (F : Filter β) : ⋂₀ (comap f F).sets = ⋂ U ∈ F, f ⁻¹' U := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter β
    ⊢ Eq (Filter.comap f F).sets.sInter (Set.iInter fun U => Set.iInter fun h => S …
  -/
  ext x
  suffices (∀ (A : Set α) (B : Set β), B ∈ F → f ⁻¹' B ⊆ A → x ∈ A) ↔
      ∀ B : Set β, B ∈ F → f x ∈ B by
    simp only [mem_sInter, mem_iInter, Filter.mem_sets, mem_comap, this, and_imp, exists_prop,
      mem_preimage, exists_imp]
  /-
    case h
    α : Type u
    β : Type v
    f : α → β
    F : Filter β
    x : α
    ⊢ Iff (∀ (A : Set α) (B : Set β), Membership.mem F B → HasSubset.Subset (Set.p …
  -/
  constructor
    /-
      case h.mp
      α : Type u
      β : Type v
      f : α → β
      F : Filter β
      x : α
      ⊢ (∀ (A : Set α) (B : Set β), Membership.mem F B → HasSubset.Subset (Set.preim …
    -/
  · intro h U U_in
    /-
      case h.mp
      α : Type u
      β : Type v
      f : α → β
      F : Filter β
      x : α
      h : ∀ (A : Set α) (B : Set β), Membership.mem F B → HasSubset.Subset (Set.prei …
      U : Set β
      U_in : Membership.mem F U
      ⊢ Membership.mem U (f x)
    -/
    simpa only [Subset.rfl, forall_prop_of_true, mem_preimage] using h (f ⁻¹' U) U U_in
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u
      β : Type v
      f : α → β
      F : Filter β
      x : α
      ⊢ (∀ (B : Set β), Membership.mem F B → Membership.mem B (f x)) → ∀ (A : Set α) …
    -/
  · intro h V U U_in f_U_V
    /-
      case h.mpr
      α : Type u
      β : Type v
      f : α → β
      F : Filter β
      x : α
      h : ∀ (B : Set β), Membership.mem F B → Membership.mem B (f x)
      V : Set α
      U : Set β
      U_in : Membership.mem F U
      f_U_V : HasSubset.Subset (Set.preimage f U) V
      ⊢ Membership.mem V x
    -/
    exact f_U_V (h U U_in)
    /-
      🎉 no goals
    -/


theorem map_iInf_le {f : ι → Filter α} {m : α → β} : map m (iInf f) ≤ ⨅ i, map m (f i) :=
  le_iInf fun _ => map_mono <| iInf_le _ _


theorem map_iInf_eq {f : ι → Filter α} {m : α → β} (hf : Directed (· ≥ ·) f) [Nonempty ι] :
    map m (iInf f) = ⨅ i, map m (f i) :=
  map_iInf_le.antisymm fun s (hs : m ⁻¹' s ∈ iInf f) =>
    let ⟨i, hi⟩ := (mem_iInf_of_directed hf _).1 hs
    have : ⨅ i, map m (f i) ≤ 𝓟 s :=
                            /-
                              α : Type u
                              β : Type v
                              ι : Sort x
                              f : ι → Filter α
                              m : α → β
                              hf : Directed (fun x1 x2 => GE.ge x1 x2) f
                              inst✝ : Nonempty ι
                              s : Set β
                              hs : Membership.mem (iInf f) (Set.preimage m s)
                              i : ι
                              hi : Membership.mem (f i) (Set.preimage m s)
                              ⊢ LE.le (Filter.map m (f i)) (Filter.principal s)
                            -/
      iInf_le_of_le i <| by simpa only [le_principal_iff, mem_map]
                            /-
                              🎉 no goals
                            -/
    Filter.le_principal_iff.1 this


theorem map_biInf_eq {ι : Type w} {f : ι → Filter α} {m : α → β} {p : ι → Prop}
    (h : DirectedOn (f ⁻¹'o (· ≥ ·)) { x | p x }) (ne : ∃ i, p i) :
    map m (⨅ (i) (_ : p i), f i) = ⨅ (i) (_ : p i), map m (f i) := by
  /-
    α : Type u
    β : Type v
    ι : Type w
    f : ι → Filter α
    m : α → β
    p : ι → Prop
    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) (setOf fun x => p x)
    ne : Exists fun i => p i
    ⊢ Eq (Filter.map m (iInf fun i => iInf fun x => f i)) (iInf fun i => iInf fun  …
  -/
  haveI := nonempty_subtype.2 ne
  /-
    α : Type u
    β : Type v
    ι : Type w
    f : ι → Filter α
    m : α → β
    p : ι → Prop
    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) (setOf fun x => p x)
    ne : Exists fun i => p i
    this : Nonempty (Subtype p)
    ⊢ Eq (Filter.map m (iInf fun i => iInf fun x => f i)) (iInf fun i => iInf fun  …
  -/
  simp only [iInf_subtype']
  /-
    α : Type u
    β : Type v
    ι : Type w
    f : ι → Filter α
    m : α → β
    p : ι → Prop
    h : DirectedOn (Order.Preimage f fun x1 x2 => GE.ge x1 x2) (setOf fun x => p x)
    ne : Exists fun i => p i
    this : Nonempty (Subtype p)
    ⊢ Eq (Filter.map m (iInf fun x => f ↑x)) (iInf fun x => Filter.map m (f ↑x))
  -/
  exact map_iInf_eq h.directed_val
  /-
    🎉 no goals
  -/


theorem map_inf_le {f g : Filter α} {m : α → β} : map m (f ⊓ g) ≤ map m f ⊓ map m g :=
  (@map_mono _ _ m).map_inf_le f g


theorem map_inf {f g : Filter α} {m : α → β} (h : Injective m) :
    map m (f ⊓ g) = map m f ⊓ map m g := by
  /-
    α : Type u
    β : Type v
    f g : Filter α
    m : α → β
    h : Function.Injective m
    ⊢ Eq (Filter.map m (Min.min f g)) (Min.min (Filter.map m f) (Filter.map m g))
  -/
  refine map_inf_le.antisymm ?_
  /-
    α : Type u
    β : Type v
    f g : Filter α
    m : α → β
    h : Function.Injective m
    ⊢ LE.le (Min.min (Filter.map m f) (Filter.map m g)) (Filter.map m (Min.min f g))
  -/
  rintro t ⟨s₁, hs₁, s₂, hs₂, ht : m ⁻¹' t = s₁ ∩ s₂⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    f g : Filter α
    m : α → β
    h : Function.Injective m
    t : Set β
    s₁ : Set α
    hs₁ : Membership.mem f s₁
    s₂ : Set α
    hs₂ : Membership.mem g s₂
    ht : Eq (Set.preimage m t) (Inter.inter s₁ s₂)
    ⊢ Membership.mem (Min.min (Filter.map m f) (Filter.map m g)) t
  -/
  refine mem_inf_of_inter (image_mem_map hs₁) (image_mem_map hs₂) ?_
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    f g : Filter α
    m : α → β
    h : Function.Injective m
    t : Set β
    s₁ : Set α
    hs₁ : Membership.mem f s₁
    s₂ : Set α
    hs₂ : Membership.mem g s₂
    ht : Eq (Set.preimage m t) (Inter.inter s₁ s₂)
    ⊢ HasSubset.Subset (Inter.inter (Set.image m s₁) (Set.image m s₂)) t
  -/
  rw [← image_inter h, image_subset_iff, ht]
  /-
    🎉 no goals
  -/


theorem map_inf' {f g : Filter α} {m : α → β} {t : Set α} (htf : t ∈ f) (htg : t ∈ g)
    (h : InjOn m t) : map m (f ⊓ g) = map m f ⊓ map m g := by
  /-
    α : Type u
    β : Type v
    f g : Filter α
    m : α → β
    t : Set α
    htf : Membership.mem f t
    htg : Membership.mem g t
    h : Set.InjOn m t
    ⊢ Eq (Filter.map m (Min.min f g)) (Min.min (Filter.map m f) (Filter.map m g))
  -/
  lift f to Filter t using htf; lift g to Filter t using htg
  /-
    case intro.intro
    α : Type u
    β : Type v
    m : α → β
    t : Set α
    h : Set.InjOn m t
    f g : Filter ↑t
    ⊢ Eq (Filter.map m (Min.min (Filter.map Subtype.val f) (Filter.map Subtype.val …
  -/
  replace h : Injective (m ∘ ((↑) : t → α)) := h.injective
  /-
    case intro.intro
    α : Type u
    β : Type v
    m : α → β
    t : Set α
    f g : Filter ↑t
    h : Function.Injective (Function.comp m Subtype.val)
    ⊢ Eq (Filter.map m (Min.min (Filter.map Subtype.val f) (Filter.map Subtype.val …
  -/
  simp only [map_map, ← map_inf Subtype.coe_injective, map_inf h]
  /-
    🎉 no goals
  -/


lemma disjoint_of_map {α β : Type*} {F G : Filter α} {f : α → β}
    (h : Disjoint (map f F) (map f G)) : Disjoint F G :=
  disjoint_iff.mpr <| map_eq_bot_iff.mp <| le_bot_iff.mp <| trans map_inf_le (disjoint_iff.mp h)


theorem disjoint_map {m : α → β} (hm : Injective m) {f₁ f₂ : Filter α} :
    Disjoint (map m f₁) (map m f₂) ↔ Disjoint f₁ f₂ := by
  /-
    α : Type u
    β : Type v
    m : α → β
    hm : Function.Injective m
    f₁ f₂ : Filter α
    ⊢ Iff (Disjoint (Filter.map m f₁) (Filter.map m f₂)) (Disjoint f₁ f₂)
  -/
  simp only [disjoint_iff, ← map_inf hm, map_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem map_equiv_symm (e : α ≃ β) (f : Filter β) : map e.symm f = comap e f :=
  map_injective e.injective <| by
    /-
      α : Type u
      β : Type v
      e : Equiv α β
      f : Filter β
      ⊢ Eq (Filter.map (⇑e) (Filter.map (⇑e.symm) f)) (Filter.map (⇑e) (Filter.comap …
    -/
    rw [map_map, e.self_comp_symm, map_id, map_comap_of_surjective e.surjective]
    /-
      🎉 no goals
    -/


theorem map_eq_comap_of_inverse {f : Filter α} {m : α → β} {n : β → α} (h₁ : m ∘ n = id)
    (h₂ : n ∘ m = id) : map m f = comap n f :=
  map_equiv_symm ⟨n, m, congr_fun h₁, congr_fun h₂⟩ f


theorem comap_equiv_symm (e : α ≃ β) (f : Filter α) : comap e.symm f = map e f :=
  (map_eq_comap_of_inverse e.self_comp_symm e.symm_comp_self).symm


theorem map_swap_eq_comap_swap {f : Filter (α × β)} : Prod.swap <$> f = comap Prod.swap f :=
  map_eq_comap_of_inverse Prod.swap_swap_eq Prod.swap_swap_eq


/-- A useful lemma when dealing with uniformities. -/
theorem map_swap4_eq_comap {f : Filter ((α × β) × γ × δ)} :
    map (fun p : (α × β) × γ × δ => ((p.1.1, p.2.1), (p.1.2, p.2.2))) f =
      comap (fun p : (α × γ) × β × δ => ((p.1.1, p.2.1), (p.1.2, p.2.2))) f :=
  map_eq_comap_of_inverse (funext fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl) (funext fun ⟨⟨_, _⟩, ⟨_, _⟩⟩ => rfl)


theorem le_map {f : Filter α} {m : α → β} {g : Filter β} (h : ∀ s ∈ f, m '' s ∈ g) : g ≤ f.map m :=
  fun _ hs => mem_of_superset (h _ hs) <| image_preimage_subset _ _


theorem le_map_iff {f : Filter α} {m : α → β} {g : Filter β} : g ≤ f.map m ↔ ∀ s ∈ f, m '' s ∈ g :=
  ⟨fun h _ hs => h (image_mem_map hs), le_map⟩


protected theorem push_pull (f : α → β) (F : Filter α) (G : Filter β) :
    map f (F ⊓ comap f G) = map f F ⊓ G := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    G : Filter β
    ⊢ Eq (Filter.map f (Min.min F (Filter.comap f G))) (Min.min (Filter.map f F) G)
  -/
  apply le_antisymm
  · calc
      map f (F ⊓ comap f G) ≤ map f F ⊓ (map f <| comap f G) := map_inf_le
      _ ≤ map f F ⊓ G := inf_le_inf_left (map f F) map_comap_le

    /-
      case a
      α : Type u
      β : Type v
      f : α → β
      F : Filter α
      G : Filter β
      ⊢ LE.le (Min.min (Filter.map f F) G) (Filter.map f (Min.min F (Filter.comap f  …
    -/
  · rintro U ⟨V, V_in, W, ⟨Z, Z_in, hZ⟩, h⟩
    /-
      case a.intro.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      f : α → β
      F : Filter α
      G : Filter β
      U : Set β
      V : Set α
      V_in : Membership.mem F V
      W : Set α
      h : Eq (Set.preimage f U) (Inter.inter V W)
      Z : Set β
      Z_in : Membership.mem G Z
      hZ : HasSubset.Subset (Set.preimage f Z) W
      ⊢ Membership.mem (Min.min (Filter.map f F) G) U
    -/
    apply mem_inf_of_inter (image_mem_map V_in) Z_in
    calc
      f '' V ∩ Z = f '' (V ∩ f ⁻¹' Z) := by rw [image_inter_preimage]
      _ ⊆ f '' (V ∩ W) := image_subset _ (inter_subset_inter_right _ ‹_›)
      _ = f '' (f ⁻¹' U) := by rw [h]
      _ ⊆ U := image_preimage_subset f U


protected theorem push_pull' (f : α → β) (F : Filter α) (G : Filter β) :
                                              /-
                                                α : Type u
                                                β : Type v
                                                f : α → β
                                                F : Filter α
                                                G : Filter β
                                                ⊢ Eq (Filter.map f (Min.min (Filter.comap f G) F)) (Min.min G (Filter.map f F))
                                              -/
    map f (comap f G ⊓ F) = G ⊓ map f F := by simp only [Filter.push_pull, inf_comm]
                                              /-
                                                🎉 no goals
                                              -/


theorem disjoint_comap_iff_map {f : α → β} {F : Filter α} {G : Filter β} :
    Disjoint F (comap f G) ↔ Disjoint (map f F) G := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    G : Filter β
    ⊢ Iff (Disjoint F (Filter.comap f G)) (Disjoint (Filter.map f F) G)
  -/
  simp only [disjoint_iff, ← Filter.push_pull, map_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem disjoint_comap_iff_map' {f : α → β} {F : Filter α} {G : Filter β} :
    Disjoint (comap f G) F ↔ Disjoint G (map f F) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    G : Filter β
    ⊢ Iff (Disjoint (Filter.comap f G) F) (Disjoint G (Filter.map f F))
  -/
  simp only [disjoint_iff, ← Filter.push_pull', map_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem neBot_inf_comap_iff_map {f : α → β} {F : Filter α} {G : Filter β} :
    NeBot (F ⊓ comap f G) ↔ NeBot (map f F ⊓ G) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    G : Filter β
    ⊢ Iff (Min.min F (Filter.comap f G)).NeBot (Min.min (Filter.map f F) G).NeBot
  -/
  rw [← map_neBot_iff, Filter.push_pull]
  /-
    🎉 no goals
  -/


theorem neBot_inf_comap_iff_map' {f : α → β} {F : Filter α} {G : Filter β} :
    NeBot (comap f G ⊓ F) ↔ NeBot (G ⊓ map f F) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    F : Filter α
    G : Filter β
    ⊢ Iff (Min.min (Filter.comap f G) F).NeBot (Min.min G (Filter.map f F)).NeBot
  -/
  rw [← map_neBot_iff, Filter.push_pull']
  /-
    🎉 no goals
  -/


theorem comap_inf_principal_neBot_of_image_mem {f : Filter β} {m : α → β} (hf : NeBot f) {s : Set α}
    (hs : m '' s ∈ f) : NeBot (comap m f ⊓ 𝓟 s) := by
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    hf : f.NeBot
    s : Set α
    hs : Membership.mem f (Set.image m s)
    ⊢ (Min.min (Filter.comap m f) (Filter.principal s)).NeBot
  -/
  rw [neBot_inf_comap_iff_map', map_principal, ← frequently_mem_iff_neBot]
  /-
    α : Type u
    β : Type v
    f : Filter β
    m : α → β
    hf : f.NeBot
    s : Set α
    hs : Membership.mem f (Set.image m s)
    ⊢ Filter.Frequently (fun x => Membership.mem (Set.image m s) x) f
  -/
  exact Eventually.frequently hs
  /-
    🎉 no goals
  -/


                                                                               /-
                                                                                 α : Type u
                                                                                 s : Set α
                                                                                 ⊢ Eq (Filter.principal s) (Filter.map Subtype.val Top.top)
                                                                               -/
theorem principal_eq_map_coe_top (s : Set α) : 𝓟 s = map ((↑) : s → α) ⊤ := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem inf_principal_eq_bot_iff_comap {F : Filter α} {s : Set α} :
    F ⊓ 𝓟 s = ⊥ ↔ comap ((↑) : s → α) F = ⊥ := by
  /-
    α : Type u
    F : Filter α
    s : Set α
    ⊢ Iff (Eq (Min.min F (Filter.principal s)) Bot.bot) (Eq (Filter.comap Subtype. …
  -/
  rw [principal_eq_map_coe_top s, ← Filter.push_pull', inf_top_eq, map_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem singleton_mem_pure {a : α} : {a} ∈ (pure a : Filter α) :=
  mem_singleton a


theorem pure_injective : Injective (pure : α → Filter α) := fun a _ hab =>
  (Filter.ext_iff.1 hab { x | a = x }).1 rfl


instance pure_neBot {α : Type u} {a : α} : NeBot (pure a) :=
  ⟨mt empty_mem_iff_bot.2 <| not_mem_empty a⟩


@[simp]
theorem le_pure_iff {f : Filter α} {a : α} : f ≤ pure a ↔ {a} ∈ f := by
  /-
    α : Type u
    f : Filter α
    a : α
    ⊢ Iff (LE.le f (Pure.pure a)) (Membership.mem f (Singleton.singleton a))
  -/
  rw [← principal_singleton, le_principal_iff]
  /-
    🎉 no goals
  -/


theorem mem_seq_def {f : Filter (α → β)} {g : Filter α} {s : Set β} :
    s ∈ f.seq g ↔ ∃ u ∈ f, ∃ t ∈ g, ∀ x ∈ u, ∀ y ∈ t, (x : α → β) y ∈ s :=
  Iff.rfl


theorem mem_seq_iff {f : Filter (α → β)} {g : Filter α} {s : Set β} :
    s ∈ f.seq g ↔ ∃ u ∈ f, ∃ t ∈ g, Set.seq u t ⊆ s := by
  /-
    α : Type u
    β : Type v
    f : Filter (α → β)
    g : Filter α
    s : Set β
    ⊢ Iff (Membership.mem (f.seq g) s) (Exists fun u => And (Membership.mem f u) ( …
  -/
  simp only [mem_seq_def, seq_subset, exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_map_seq_iff {f : Filter α} {g : Filter β} {m : α → β → γ} {s : Set γ} :
    s ∈ (f.map m).seq g ↔ ∃ t u, t ∈ g ∧ u ∈ f ∧ ∀ x ∈ u, ∀ y ∈ t, m x y ∈ s :=
  Iff.intro (fun ⟨t, ht, s, hs, hts⟩ => ⟨s, m ⁻¹' t, hs, ht, fun _ => hts _⟩)
    fun ⟨t, s, ht, hs, hts⟩ =>
    ⟨m '' s, image_mem_map hs, t, ht, fun _ ⟨_, has, Eq⟩ => Eq ▸ hts _ has⟩


theorem seq_mem_seq {f : Filter (α → β)} {g : Filter α} {s : Set (α → β)} {t : Set α} (hs : s ∈ f)
    (ht : t ∈ g) : s.seq t ∈ f.seq g :=
  ⟨s, hs, t, ht, fun f hf a ha => ⟨f, hf, a, ha, rfl⟩⟩


theorem le_seq {f : Filter (α → β)} {g : Filter α} {h : Filter β}
    (hh : ∀ t ∈ f, ∀ u ∈ g, Set.seq t u ∈ h) : h ≤ seq f g := fun _ ⟨_, ht, _, hu, hs⟩ =>
  mem_of_superset (hh _ ht _ hu) fun _ ⟨_, hm, _, ha, eq⟩ => eq ▸ hs _ hm _ ha


@[mono]
theorem seq_mono {f₁ f₂ : Filter (α → β)} {g₁ g₂ : Filter α} (hf : f₁ ≤ f₂) (hg : g₁ ≤ g₂) :
    f₁.seq g₁ ≤ f₂.seq g₂ :=
  le_seq fun _ hs _ ht => seq_mem_seq (hf hs) (hg ht)


@[simp]
theorem pure_seq_eq_map (g : α → β) (f : Filter α) : seq (pure g) f = f.map g := by
  /-
    α : Type u
    β : Type v
    g : α → β
    f : Filter α
    ⊢ Eq ((Pure.pure g).seq f) (Filter.map g f)
  -/
  refine le_antisymm (le_map fun s hs => ?_) (le_seq fun s hs t ht => ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      ⊢ Membership.mem ((Pure.pure g).seq f) (Set.image g s)
    -/
  · rw [← singleton_seq]
    /-
      case refine_1
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      ⊢ Membership.mem ((Pure.pure g).seq f) ((Singleton.singleton g).seq s)
    -/
    apply seq_mem_seq _ hs
    /-
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set α
      hs : Membership.mem f s
      ⊢ Membership.mem (Pure.pure g) (Singleton.singleton g)
    -/
    exact singleton_mem_pure
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set (α → β)
      hs : Membership.mem (Pure.pure g) s
      t : Set α
      ht : Membership.mem f t
      ⊢ Membership.mem (Filter.map g f) (s.seq t)
    -/
  · refine sets_of_superset (map g f) (image_mem_map ht) ?_
    /-
      case refine_2
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set (α → β)
      hs : Membership.mem (Pure.pure g) s
      t : Set α
      ht : Membership.mem f t
      ⊢ HasSubset.Subset (Set.image g t) (s.seq t)
    -/
    rintro b ⟨a, ha, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u
      β : Type v
      g : α → β
      f : Filter α
      s : Set (α → β)
      hs : Membership.mem (Pure.pure g) s
      t : Set α
      ht : Membership.mem f t
      a : α
      ha : Membership.mem t a
      ⊢ Membership.mem (s.seq t) (g a)
    -/
    exact ⟨g, hs, a, ha, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem seq_pure (f : Filter (α → β)) (a : α) : seq f (pure a) = map (fun g : α → β => g a) f := by
  /-
    α : Type u
    β : Type v
    f : Filter (α → β)
    a : α
    ⊢ Eq (f.seq (Pure.pure a)) (Filter.map (fun g => g a) f)
  -/
  refine le_antisymm (le_map fun s hs => ?_) (le_seq fun s hs t ht => ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      f : Filter (α → β)
      a : α
      s : Set (α → β)
      hs : Membership.mem f s
      ⊢ Membership.mem (f.seq (Pure.pure a)) (Set.image (fun g => g a) s)
    -/
  · rw [← seq_singleton]
    /-
      case refine_1
      α : Type u
      β : Type v
      f : Filter (α → β)
      a : α
      s : Set (α → β)
      hs : Membership.mem f s
      ⊢ Membership.mem (f.seq (Pure.pure a)) (s.seq (Singleton.singleton a))
    -/
    exact seq_mem_seq hs singleton_mem_pure
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      f : Filter (α → β)
      a : α
      s : Set (α → β)
      hs : Membership.mem f s
      t : Set α
      ht : Membership.mem (Pure.pure a) t
      ⊢ Membership.mem (Filter.map (fun g => g a) f) (s.seq t)
    -/
  · refine sets_of_superset (map (fun g : α → β => g a) f) (image_mem_map hs) ?_
    /-
      case refine_2
      α : Type u
      β : Type v
      f : Filter (α → β)
      a : α
      s : Set (α → β)
      hs : Membership.mem f s
      t : Set α
      ht : Membership.mem (Pure.pure a) t
      ⊢ HasSubset.Subset (Set.image (fun g => g a) s) (s.seq t)
    -/
    rintro b ⟨g, hg, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u
      β : Type v
      f : Filter (α → β)
      a : α
      s : Set (α → β)
      hs : Membership.mem f s
      t : Set α
      ht : Membership.mem (Pure.pure a) t
      g : α → β
      hg : Membership.mem s g
      ⊢ Membership.mem (s.seq t) ((fun g => g a) g)
    -/
    exact ⟨g, hg, a, ht, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem seq_assoc (x : Filter α) (g : Filter (α → β)) (h : Filter (β → γ)) :
    seq h (seq g x) = seq (seq (map (· ∘ ·) h) g) x := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    x : Filter α
    g : Filter (α → β)
    h : Filter (β → γ)
    ⊢ Eq (h.seq (g.seq x)) (((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq …
  -/
  refine le_antisymm (le_seq fun s hs t ht => ?_) (le_seq fun s hs t ht => ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (α → γ)
      hs : Membership.mem ((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g) s
      t : Set α
      ht : Membership.mem x t
      ⊢ Membership.mem (h.seq (g.seq x)) (s.seq t)
    -/
  · rcases mem_seq_iff.1 hs with ⟨u, hu, v, hv, hs⟩
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (α → γ)
      hs✝ : Membership.mem ((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g) s
      t : Set α
      ht : Membership.mem x t
      u : Set ((α → β) → α → γ)
      hu : Membership.mem (Filter.map (fun x1 x2 => Function.comp x1 x2) h) u
      v : Set (α → β)
      hv : Membership.mem g v
      hs : HasSubset.Subset (u.seq v) s
      ⊢ Membership.mem (h.seq (g.seq x)) (s.seq t)
    -/
    rcases mem_map_iff_exists_image.1 hu with ⟨w, hw, hu⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (α → γ)
      hs✝ : Membership.mem ((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g) s
      t : Set α
      ht : Membership.mem x t
      u : Set ((α → β) → α → γ)
      hu✝ : Membership.mem (Filter.map (fun x1 x2 => Function.comp x1 x2) h) u
      v : Set (α → β)
      hv : Membership.mem g v
      hs : HasSubset.Subset (u.seq v) s
      w : Set (β → γ)
      hw : Membership.mem h w
      hu : HasSubset.Subset (Set.image (fun x1 x2 => Function.comp x1 x2) w) u
      ⊢ Membership.mem (h.seq (g.seq x)) (s.seq t)
    -/
    refine mem_of_superset ?_ (Set.seq_mono ((Set.seq_mono hu Subset.rfl).trans hs) Subset.rfl)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (α → γ)
      hs✝ : Membership.mem ((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g) s
      t : Set α
      ht : Membership.mem x t
      u : Set ((α → β) → α → γ)
      hu✝ : Membership.mem (Filter.map (fun x1 x2 => Function.comp x1 x2) h) u
      v : Set (α → β)
      hv : Membership.mem g v
      hs : HasSubset.Subset (u.seq v) s
      w : Set (β → γ)
      hw : Membership.mem h w
      hu : HasSubset.Subset (Set.image (fun x1 x2 => Function.comp x1 x2) w) u
      ⊢ Membership.mem (h.seq (g.seq x)) (((Set.image (fun x1 x2 => Function.comp x1 …
    -/
    rw [← Set.seq_seq]
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (α → γ)
      hs✝ : Membership.mem ((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g) s
      t : Set α
      ht : Membership.mem x t
      u : Set ((α → β) → α → γ)
      hu✝ : Membership.mem (Filter.map (fun x1 x2 => Function.comp x1 x2) h) u
      v : Set (α → β)
      hv : Membership.mem g v
      hs : HasSubset.Subset (u.seq v) s
      w : Set (β → γ)
      hw : Membership.mem h w
      hu : HasSubset.Subset (Set.image (fun x1 x2 => Function.comp x1 x2) w) u
      ⊢ Membership.mem (h.seq (g.seq x)) (w.seq (v.seq t))
    -/
    exact seq_mem_seq hw (seq_mem_seq hv ht)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (β → γ)
      hs : Membership.mem h s
      t : Set β
      ht : Membership.mem (g.seq x) t
      ⊢ Membership.mem (((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g).se …
    -/
  · rcases mem_seq_iff.1 ht with ⟨u, hu, v, hv, ht⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (β → γ)
      hs : Membership.mem h s
      t : Set β
      ht✝ : Membership.mem (g.seq x) t
      u : Set (α → β)
      hu : Membership.mem g u
      v : Set α
      hv : Membership.mem x v
      ht : HasSubset.Subset (u.seq v) t
      ⊢ Membership.mem (((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g).se …
    -/
    refine mem_of_superset ?_ (Set.seq_mono Subset.rfl ht)
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (β → γ)
      hs : Membership.mem h s
      t : Set β
      ht✝ : Membership.mem (g.seq x) t
      u : Set (α → β)
      hu : Membership.mem g u
      v : Set α
      hv : Membership.mem x v
      ht : HasSubset.Subset (u.seq v) t
      ⊢ Membership.mem (((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g).se …
    -/
    rw [Set.seq_seq]
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u
      β : Type v
      γ : Type w
      x : Filter α
      g : Filter (α → β)
      h : Filter (β → γ)
      s : Set (β → γ)
      hs : Membership.mem h s
      t : Set β
      ht✝ : Membership.mem (g.seq x) t
      u : Set (α → β)
      hu : Membership.mem g u
      v : Set α
      hv : Membership.mem x v
      ht : HasSubset.Subset (u.seq v) t
      ⊢ Membership.mem (((Filter.map (fun x1 x2 => Function.comp x1 x2) h).seq g).se …
    -/
    exact seq_mem_seq (seq_mem_seq (image_mem_map hs) hu) hv
    /-
      🎉 no goals
    -/


theorem prod_map_seq_comm (f : Filter α) (g : Filter β) :
    (map Prod.mk f).seq g = seq (map (fun b a => (a, b)) g) f := by
  /-
    α : Type u
    β : Type v
    f : Filter α
    g : Filter β
    ⊢ Eq ((Filter.map Prod.mk f).seq g) ((Filter.map (fun b a => { fst := a, snd : …
  -/
  refine le_antisymm (le_seq fun s hs t ht => ?_) (le_seq fun s hs t ht => ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (α → Prod α β)
      hs : Membership.mem (Filter.map (fun b a => { fst := a, snd := b }) g) s
      t : Set α
      ht : Membership.mem f t
      ⊢ Membership.mem ((Filter.map Prod.mk f).seq g) (s.seq t)
    -/
  · rcases mem_map_iff_exists_image.1 hs with ⟨u, hu, hs⟩
    /-
      case refine_1.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (α → Prod α β)
      hs✝ : Membership.mem (Filter.map (fun b a => { fst := a, snd := b }) g) s
      t : Set α
      ht : Membership.mem f t
      u : Set β
      hu : Membership.mem g u
      hs : HasSubset.Subset (Set.image (fun b a => { fst := a, snd := b }) u) s
      ⊢ Membership.mem ((Filter.map Prod.mk f).seq g) (s.seq t)
    -/
    refine mem_of_superset ?_ (Set.seq_mono hs Subset.rfl)
    /-
      case refine_1.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (α → Prod α β)
      hs✝ : Membership.mem (Filter.map (fun b a => { fst := a, snd := b }) g) s
      t : Set α
      ht : Membership.mem f t
      u : Set β
      hu : Membership.mem g u
      hs : HasSubset.Subset (Set.image (fun b a => { fst := a, snd := b }) u) s
      ⊢ Membership.mem ((Filter.map Prod.mk f).seq g) ((Set.image (fun b a => { fst  …
    -/
    rw [← Set.prod_image_seq_comm]
    /-
      case refine_1.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (α → Prod α β)
      hs✝ : Membership.mem (Filter.map (fun b a => { fst := a, snd := b }) g) s
      t : Set α
      ht : Membership.mem f t
      u : Set β
      hu : Membership.mem g u
      hs : HasSubset.Subset (Set.image (fun b a => { fst := a, snd := b }) u) s
      ⊢ Membership.mem ((Filter.map Prod.mk f).seq g) ((Set.image Prod.mk t).seq u)
    -/
    exact seq_mem_seq (image_mem_map ht) hu
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (β → Prod α β)
      hs : Membership.mem (Filter.map Prod.mk f) s
      t : Set β
      ht : Membership.mem g t
      ⊢ Membership.mem ((Filter.map (fun b a => { fst := a, snd := b }) g).seq f) (s …
    -/
  · rcases mem_map_iff_exists_image.1 hs with ⟨u, hu, hs⟩
    /-
      case refine_2.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (β → Prod α β)
      hs✝ : Membership.mem (Filter.map Prod.mk f) s
      t : Set β
      ht : Membership.mem g t
      u : Set α
      hu : Membership.mem f u
      hs : HasSubset.Subset (Set.image Prod.mk u) s
      ⊢ Membership.mem ((Filter.map (fun b a => { fst := a, snd := b }) g).seq f) (s …
    -/
    refine mem_of_superset ?_ (Set.seq_mono hs Subset.rfl)
    /-
      case refine_2.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (β → Prod α β)
      hs✝ : Membership.mem (Filter.map Prod.mk f) s
      t : Set β
      ht : Membership.mem g t
      u : Set α
      hu : Membership.mem f u
      hs : HasSubset.Subset (Set.image Prod.mk u) s
      ⊢ Membership.mem ((Filter.map (fun b a => { fst := a, snd := b }) g).seq f) (( …
    -/
    rw [Set.prod_image_seq_comm]
    /-
      case refine_2.intro.intro
      α : Type u
      β : Type v
      f : Filter α
      g : Filter β
      s : Set (β → Prod α β)
      hs✝ : Membership.mem (Filter.map Prod.mk f) s
      t : Set β
      ht : Membership.mem g t
      u : Set α
      hu : Membership.mem f u
      hs : HasSubset.Subset (Set.image Prod.mk u) s
      ⊢ Membership.mem ((Filter.map (fun b a => { fst := a, snd := b }) g).seq f) (( …
    -/
    exact seq_mem_seq (image_mem_map ht) hu
    /-
      🎉 no goals
    -/


theorem seq_eq_filter_seq {α β : Type u} (f : Filter (α → β)) (g : Filter α) :
    f <*> g = seq f g :=
  rfl


instance : LawfulApplicative (Filter : Type u → Type u) where
  map_pure := map_pure
  seqLeft_eq _ _ := rfl
  seqRight_eq _ _ := rfl
  seq_pure := seq_pure
  pure_seq := pure_seq_eq_map
  seq_assoc := seq_assoc


instance : CommApplicative (Filter : Type u → Type u) :=
  ⟨fun f g => prod_map_seq_comm f g⟩


@[simp]
theorem eventually_bind {f : Filter α} {m : α → Filter β} {p : β → Prop} :
    (∀ᶠ y in bind f m, p y) ↔ ∀ᶠ x in f, ∀ᶠ y in m x, p y :=
  Iff.rfl


@[simp]
theorem eventuallyEq_bind {f : Filter α} {m : α → Filter β} {g₁ g₂ : β → γ} :
    g₁ =ᶠ[bind f m] g₂ ↔ ∀ᶠ x in f, g₁ =ᶠ[m x] g₂ :=
  Iff.rfl


@[simp]
theorem eventuallyLE_bind [LE γ] {f : Filter α} {m : α → Filter β} {g₁ g₂ : β → γ} :
    g₁ ≤ᶠ[bind f m] g₂ ↔ ∀ᶠ x in f, g₁ ≤ᶠ[m x] g₂ :=
  Iff.rfl


theorem mem_bind' {s : Set β} {f : Filter α} {m : α → Filter β} :
    s ∈ bind f m ↔ { a | s ∈ m a } ∈ f :=
  Iff.rfl


@[simp]
theorem mem_bind {s : Set β} {f : Filter α} {m : α → Filter β} :
    s ∈ bind f m ↔ ∃ t ∈ f, ∀ x ∈ t, s ∈ m x :=
  calc
    s ∈ bind f m ↔ { a | s ∈ m a } ∈ f := Iff.rfl
    _ ↔ ∃ t ∈ f, t ⊆ { a | s ∈ m a } := exists_mem_subset_iff.symm
    _ ↔ ∃ t ∈ f, ∀ x ∈ t, s ∈ m x := Iff.rfl


theorem bind_le {f : Filter α} {g : α → Filter β} {l : Filter β} (h : ∀ᶠ x in f, g x ≤ l) :
    f.bind g ≤ l :=
  join_le <| eventually_map.2 h


@[mono]
theorem bind_mono {f₁ f₂ : Filter α} {g₁ g₂ : α → Filter β} (hf : f₁ ≤ f₂) (hg : g₁ ≤ᶠ[f₁] g₂) :
    bind f₁ g₁ ≤ bind f₂ g₂ := by
  /-
    α : Type u
    β : Type v
    f₁ f₂ : Filter α
    g₁ g₂ : α → Filter β
    hf : LE.le f₁ f₂
    hg : f₁.EventuallyLE g₁ g₂
    ⊢ LE.le (f₁.bind g₁) (f₂.bind g₂)
  -/
  refine le_trans (fun s hs => ?_) (join_mono <| map_mono hf)
  /-
    α : Type u
    β : Type v
    f₁ f₂ : Filter α
    g₁ g₂ : α → Filter β
    hf : LE.le f₁ f₂
    hg : f₁.EventuallyLE g₁ g₂
    s : Set β
    hs : Membership.mem (Filter.map g₂ f₁).join s
    ⊢ Membership.mem (f₁.bind g₁) s
  -/
  simp only [mem_join, mem_bind', mem_map] at hs ⊢
  /-
    α : Type u
    β : Type v
    f₁ f₂ : Filter α
    g₁ g₂ : α → Filter β
    hf : LE.le f₁ f₂
    hg : f₁.EventuallyLE g₁ g₂
    s : Set β
    hs : Membership.mem f₁ (Set.preimage g₂ (setOf fun t => Membership.mem t s))
    ⊢ Membership.mem f₁ (setOf fun a => Membership.mem (g₁ a) s)
  -/
  filter_upwards [hg, hs] with _ hx hs using hx hs
  /-
    🎉 no goals
  -/


theorem bind_inf_principal {f : Filter α} {g : α → Filter β} {s : Set β} :
    (f.bind fun x => g x ⊓ 𝓟 s) = f.bind g ⊓ 𝓟 s :=
                         /-
                           α : Type u
                           β : Type v
                           f : Filter α
                           g : α → Filter β
                           s✝ s : Set β
                           ⊢ Iff (Membership.mem (f.bind fun x => Min.min (g x) (Filter.principal s✝)) s) …
                         -/
  Filter.ext fun s => by simp only [mem_bind, mem_inf_principal]
                         /-
                           🎉 no goals
                         -/


theorem sup_bind {f g : Filter α} {h : α → Filter β} : bind (f ⊔ g) h = bind f h ⊔ bind g h := rfl


theorem principal_bind {s : Set α} {f : α → Filter β} : bind (𝓟 s) f = ⨆ x ∈ s, f x :=
  show join (map f (𝓟 s)) = ⨆ x ∈ s, f x by
    /-
      α : Type u
      β : Type v
      s : Set α
      f : α → Filter β
      ⊢ Eq (Filter.map f (Filter.principal s)).join (iSup fun x => iSup fun h => f x)
    -/
    simp only [sSup_image, join_principal_eq_sSup, map_principal, eq_self_iff_true]
    /-
      🎉 no goals
    -/


theorem Set.EqOn.eventuallyEq {α β} {s : Set α} {f g : α → β} (h : EqOn f g s) : f =ᶠ[𝓟 s] g :=
  h


theorem Set.EqOn.eventuallyEq_of_mem {α β} {s : Set α} {l : Filter α} {f g : α → β} (h : EqOn f g s)
    (hl : s ∈ l) : f =ᶠ[l] g :=
  h.eventuallyEq.filter_mono <| Filter.le_principal_iff.2 hl


theorem HasSubset.Subset.eventuallyLE {α} {l : Filter α} {s t : Set α} (h : s ⊆ t) : s ≤ᶠ[l] t :=
  Filter.Eventually.of_forall h


theorem Filter.map_surjOn_Iic_iff_le_map {m : α → β} :
    SurjOn (map m) (Iic F) (Iic G) ↔ G ≤ map m F := by
  /-
    α : Type u_1
    β : Type u_2
    F : Filter α
    G : Filter β
    m : α → β
    ⊢ Iff (Set.SurjOn (Filter.map m) (Set.Iic F) (Set.Iic G)) (LE.le G (Filter.map …
  -/
  refine ⟨fun hm ↦ ?_, fun hm ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      F : Filter α
      G : Filter β
      m : α → β
      hm : Set.SurjOn (Filter.map m) (Set.Iic F) (Set.Iic G)
      ⊢ LE.le G (Filter.map m F)
    -/
  · rcases hm right_mem_Iic with ⟨H, (hHF : H ≤ F), rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      F : Filter α
      m : α → β
      H : Filter α
      hHF : LE.le H F
      hm : Set.SurjOn (Filter.map m) (Set.Iic F) (Set.Iic (Filter.map m H))
      ⊢ LE.le (Filter.map m H) (Filter.map m F)
    -/
    exact map_mono hHF
    /-
      🎉 no goals
    -/
  · have : RightInvOn (F ⊓ comap m ·) (map m) (Iic G) :=
      fun H (hHG : H ≤ G) ↦ by simpa [Filter.push_pull] using hHG.trans hm
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      F : Filter α
      G : Filter β
      m : α → β
      hm : LE.le G (Filter.map m F)
      this : Set.RightInvOn (fun x => Min.min F (Filter.comap m x)) (Filter.map m) ( …
      ⊢ Set.SurjOn (Filter.map m) (Set.Iic F) (Set.Iic G)
    -/
    exact this.surjOn fun H _ ↦ mem_Iic.mpr inf_le_left
    /-
      🎉 no goals
    -/


theorem Filter.map_surjOn_Iic_iff_surjOn {s : Set α} {t : Set β} {m : α → β} :
    SurjOn (map m) (Iic <| 𝓟 s) (Iic <| 𝓟 t) ↔ SurjOn m s t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    m : α → β
    ⊢ Iff (Set.SurjOn (Filter.map m) (Set.Iic (Filter.principal s)) (Set.Iic (Filt …
  -/
  rw [map_surjOn_Iic_iff_le_map, map_principal, principal_mono, SurjOn]
  /-
    🎉 no goals
  -/


alias ⟨_, Set.SurjOn.filter_map_Iic⟩ := Filter.map_surjOn_Iic_iff_surjOn


theorem Filter.filter_injOn_Iic_iff_injOn {s : Set α} {m : α → β} :
    InjOn (map m) (Iic <| 𝓟 s) ↔ InjOn m s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    m : α → β
    ⊢ Iff (Set.InjOn (Filter.map m) (Set.Iic (Filter.principal s))) (Set.InjOn m s)
  -/
  refine ⟨fun hm x hx y hy hxy ↦ ?_, fun hm F hF G hG ↦ ?_⟩
  · rwa [← pure_injective.eq_iff, ← map_pure, ← map_pure, hm.eq_iff, pure_injective.eq_iff]
                 /-
                   case refine_1.hx
                   α : Type u_1
                   β : Type u_2
                   s : Set α
                   m : α → β
                   hm : Set.InjOn (Filter.map m) (Set.Iic (Filter.principal s))
                   x : α
                   hx : Membership.mem s x
                   y : α
                   hy : Membership.mem s y
                   hxy : Eq (Filter.map m (Pure.pure x)) (Filter.map m (Pure.pure y))
                   ⊢ Membership.mem (Set.Iic (Filter.principal s)) (Pure.pure x)
                 -/
                 /-
                   🎉 no goals
                 -/
      at hxy <;> rwa [mem_Iic, pure_le_principal]
                 /-
                   🎉 no goals
                 -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s : Set α
      m : α → β
      hm : Set.InjOn m s
      F : Filter α
      hF : Membership.mem (Set.Iic (Filter.principal s)) F
      G : Filter α
      hG : Membership.mem (Set.Iic (Filter.principal s)) G
      ⊢ Eq (Filter.map m F) (Filter.map m G) → Eq F G
    -/
  · simp [map_eq_map_iff_of_injOn (le_principal_iff.mp hF) (le_principal_iff.mp hG) hm]
    /-
      🎉 no goals
    -/


alias ⟨_, Set.InjOn.filter_map_Iic⟩ := Filter.filter_injOn_Iic_iff_injOn


lemma compl_mem_comk {p : Set α → Prop} {he hmono hunion s} :
    sᶜ ∈ comk p he hmono hunion ↔ p s := by
  /-
    α : Type u_1
    p : Set α → Prop
    he : p EmptyCollection.emptyCollection
    hmono : ∀ (t : Set α), p t → ∀ (s : Set α), HasSubset.Subset s t → p s
    hunion : ∀ (s : Set α), p s → ∀ (t : Set α), p t → p (Union.union s t)
    s : Set α
    ⊢ Iff (Membership.mem (Filter.comk p he hmono hunion) (HasCompl.compl s)) (p s)
  -/
  simp
  /-
    🎉 no goals
  -/


