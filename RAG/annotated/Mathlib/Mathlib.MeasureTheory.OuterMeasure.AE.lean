/-- The “almost everywhere” filter of co-null sets. -/
def ae (μ : F) : Filter α :=
  .ofCountableUnion (μ · = 0) (fun _S hSc ↦ (measure_sUnion_null_iff hSc).2) fun _t ht _s hs ↦
    measure_mono_null hs ht


/-- `∀ᵐ a ∂μ, p a` means that `p a` for a.e. `a`, i.e. `p` holds true away from a null set.

This is notation for `Filter.Eventually p (MeasureTheory.ae μ)`. -/
notation3 "∀ᵐ "(...)" ∂"μ", "r:(scoped p => Filter.Eventually p <| MeasureTheory.ae μ) => r


/-- `∃ᵐ a ∂μ, p a` means that `p` holds `∂μ`-frequently,
i.e. `p` holds on a set of positive measure.

This is notation for `Filter.Frequently p (MeasureTheory.ae μ)`. -/
notation3 "∃ᵐ "(...)" ∂"μ", "r:(scoped P => Filter.Frequently P <| MeasureTheory.ae μ) => r


/-- `f =ᵐ[μ] g` means `f` and `g` are eventually equal along the a.e. filter,
i.e. `f=g` away from a null set.

This is notation for `Filter.EventuallyEq (MeasureTheory.ae μ) f g`. -/
notation:50 f " =ᵐ[" μ:50 "] " g:50 => Filter.EventuallyEq (MeasureTheory.ae μ) f g


/-- `f ≤ᵐ[μ] g` means `f` is eventually less than `g` along the a.e. filter,
i.e. `f ≤ g` away from a null set.

This is notation for `Filter.EventuallyLE (MeasureTheory.ae μ) f g`. -/
notation:50 f " ≤ᵐ[" μ:50 "] " g:50 => Filter.EventuallyLE (MeasureTheory.ae μ) f g


theorem mem_ae_iff {s : Set α} : s ∈ ae μ ↔ μ sᶜ = 0 :=
  Iff.rfl


theorem ae_iff {p : α → Prop} : (∀ᵐ a ∂μ, p a) ↔ μ { a | ¬p a } = 0 :=
  Iff.rfl


                                                                 /-
                                                                   α : Type u_1
                                                                   F : Type u_3
                                                                   inst✝¹ : FunLike F (Set α) ENNReal
                                                                   inst✝ : MeasureTheory.OuterMeasureClass F α
                                                                   μ : F
                                                                   s : Set α
                                                                   ⊢ Iff (Membership.mem (MeasureTheory.ae μ) (HasCompl.compl s)) (Eq (μ s) 0)
                                                                 -/
theorem compl_mem_ae_iff {s : Set α} : sᶜ ∈ ae μ ↔ μ s = 0 := by simp only [mem_ae_iff, compl_compl]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem frequently_ae_iff {p : α → Prop} : (∃ᵐ a ∂μ, p a) ↔ μ { a | p a } ≠ 0 :=
  not_congr compl_mem_ae_iff


theorem frequently_ae_mem_iff {s : Set α} : (∃ᵐ a ∂μ, a ∈ s) ↔ μ s ≠ 0 :=
  not_congr compl_mem_ae_iff


theorem measure_zero_iff_ae_nmem {s : Set α} : μ s = 0 ↔ ∀ᵐ a ∂μ, a ∉ s :=
  compl_mem_ae_iff.symm


theorem ae_of_all {p : α → Prop} (μ : F) : (∀ a, p a) → ∀ᵐ a ∂μ, p a :=
  Eventually.of_forall


instance instCountableInterFilter : CountableInterFilter (ae μ) := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ CountableInterFilter (MeasureTheory.ae μ)
  -/
  unfold ae; infer_instance
             /-
               🎉 no goals
             -/


theorem ae_all_iff {ι : Sort*} [Countable ι] {p : α → ι → Prop} :
    (∀ᵐ a ∂μ, ∀ i, p a i) ↔ ∀ i, ∀ᵐ a ∂μ, p a i :=
  eventually_countable_forall


theorem all_ae_of {ι : Sort*} {p : α → ι → Prop} (hp : ∀ᵐ a ∂μ, ∀ i, p a i) (i : ι) :
    ∀ᵐ a ∂μ, p a i := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    ι : Sort u_4
    p : α → ι → Prop
    hp : Filter.Eventually (fun a => ∀ (i : ι), p a i) (MeasureTheory.ae μ)
    i : ι
    ⊢ Filter.Eventually (fun a => p a i) (MeasureTheory.ae μ)
  -/
  filter_upwards [hp] with a ha using ha i
  /-
    🎉 no goals
  -/


lemma ae_iff_of_countable [Countable α] {p : α → Prop} : (∀ᵐ x ∂μ, p x) ↔ ∀ x, μ {x} ≠ 0 → p x := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝ : Countable α
    p : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (MeasureTheory.ae μ)) (∀ (x : α), Ne ( …
  -/
  rw [ae_iff, measure_null_iff_singleton]
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    μ : F
    inst✝ : Countable α
    p : α → Prop
    ⊢ Iff (∀ (x : α), Membership.mem (setOf fun a => Not (p a)) x → Eq (μ (Singlet …
  -/
  exacts [forall_congr' fun _ ↦ not_imp_comm, Set.to_countable _]
  /-
    🎉 no goals
  -/


theorem ae_ball_iff {ι : Type*} {S : Set ι} (hS : S.Countable) {p : α → ∀ i ∈ S, Prop} :
    (∀ᵐ x ∂μ, ∀ i (hi : i ∈ S), p x i hi) ↔ ∀ i (hi : i ∈ S), ∀ᵐ x ∂μ, p x i hi :=
  eventually_countable_ball hS


lemma ae_eq_refl (f : α → β) : f =ᵐ[μ] f := EventuallyEq.rfl

lemma ae_eq_rfl {f : α → β} : f =ᵐ[μ] f := EventuallyEq.rfl

lemma ae_eq_comm {f g : α → β} : f =ᵐ[μ] g ↔ g =ᵐ[μ] f := eventuallyEq_comm


theorem ae_eq_symm {f g : α → β} (h : f =ᵐ[μ] g) : g =ᵐ[μ] f :=
  h.symm


theorem ae_eq_trans {f g h : α → β} (h₁ : f =ᵐ[μ] g) (h₂ : g =ᵐ[μ] h) : f =ᵐ[μ] h :=
  h₁.trans h₂


@[simp] lemma ae_eq_top  : ae μ = ⊤ ↔ ∀ a, μ {a} ≠ 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    ⊢ Iff (Eq (MeasureTheory.ae μ) Top.top) (∀ (a : α), Ne (μ (Singleton.singleton …
  -/
  simp only [Filter.ext_iff, mem_ae_iff, mem_top, ne_eq]
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    ⊢ Iff (∀ (s : Set α), Iff (Eq (μ (HasCompl.compl s)) 0) (Eq s Set.univ)) (∀ (a …
  -/
  refine ⟨fun h a ha ↦ by simpa [ha] using (h {a}ᶜ).1, fun h s ↦ ⟨fun hs ↦ ?_, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      F : Type u_3
      inst✝¹ : FunLike F (Set α) ENNReal
      inst✝ : MeasureTheory.OuterMeasureClass F α
      μ : F
      h : ∀ (a : α), Not (Eq (μ (Singleton.singleton a)) 0)
      s : Set α
      hs : Eq (μ (HasCompl.compl s)) 0
      ⊢ Eq s Set.univ
    -/
  · rw [← compl_empty_iff, ← not_nonempty_iff_eq_empty]
    /-
      case refine_1
      α : Type u_1
      F : Type u_3
      inst✝¹ : FunLike F (Set α) ENNReal
      inst✝ : MeasureTheory.OuterMeasureClass F α
      μ : F
      h : ∀ (a : α), Not (Eq (μ (Singleton.singleton a)) 0)
      s : Set α
      hs : Eq (μ (HasCompl.compl s)) 0
      ⊢ Not (HasCompl.compl s).Nonempty
    -/
    rintro ⟨a, ha⟩
    /-
      case refine_1.intro
      α : Type u_1
      F : Type u_3
      inst✝¹ : FunLike F (Set α) ENNReal
      inst✝ : MeasureTheory.OuterMeasureClass F α
      μ : F
      h : ∀ (a : α), Not (Eq (μ (Singleton.singleton a)) 0)
      s : Set α
      hs : Eq (μ (HasCompl.compl s)) 0
      a : α
      ha : Membership.mem (HasCompl.compl s) a
      ⊢ False
    -/
    exact h _ <| measure_mono_null (singleton_subset_iff.2 ha) hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F : Type u_3
      inst✝¹ : FunLike F (Set α) ENNReal
      inst✝ : MeasureTheory.OuterMeasureClass F α
      μ : F
      h : ∀ (a : α), Not (Eq (μ (Singleton.singleton a)) 0)
      s : Set α
      ⊢ Eq s Set.univ → Eq (μ (HasCompl.compl s)) 0
    -/
  · rintro rfl
    /-
      case refine_2
      α : Type u_1
      F : Type u_3
      inst✝¹ : FunLike F (Set α) ENNReal
      inst✝ : MeasureTheory.OuterMeasureClass F α
      μ : F
      h : ∀ (a : α), Not (Eq (μ (Singleton.singleton a)) 0)
      ⊢ Eq (μ (HasCompl.compl Set.univ)) 0
    -/
    simp
    /-
      🎉 no goals
    -/


theorem ae_le_of_ae_lt {β : Type*} [Preorder β] {f g : α → β} (h : ∀ᵐ x ∂μ, f x < g x) :
    f ≤ᵐ[μ] g :=
  h.mono fun _ ↦ le_of_lt


@[simp]
theorem ae_eq_empty : s =ᵐ[μ] (∅ : Set α) ↔ μ s = 0 :=
                                 /-
                                   α : Type u_1
                                   F : Type u_3
                                   inst✝¹ : FunLike F (Set α) ENNReal
                                   inst✝ : MeasureTheory.OuterMeasureClass F α
                                   μ : F
                                   s : Set α
                                   ⊢ Iff (Filter.Eventually (fun x => Not (Membership.mem s x)) (MeasureTheory.ae …
                                 -/
  eventuallyEq_empty.trans <| by simp only [ae_iff, Classical.not_not, setOf_mem_eq]
                                 /-
                                   🎉 no goals
                                 -/

-- Porting note: The priority should be higher than `eventuallyEq_univ`.

@[simp high]
theorem ae_eq_univ : s =ᵐ[μ] (univ : Set α) ↔ μ sᶜ = 0 :=
  eventuallyEq_univ


theorem ae_le_set : s ≤ᵐ[μ] t ↔ μ (s \ t) = 0 :=
  calc
    s ≤ᵐ[μ] t ↔ ∀ᵐ x ∂μ, x ∈ s → x ∈ t := Iff.rfl
                            /-
                              α : Type u_1
                              F : Type u_3
                              inst✝¹ : FunLike F (Set α) ENNReal
                              inst✝ : MeasureTheory.OuterMeasureClass F α
                              μ : F
                              s t : Set α
                              ⊢ Iff (Filter.Eventually (fun x => Membership.mem s x → Membership.mem t x) (M …
                            -/
    _ ↔ μ (s \ t) = 0 := by simp [ae_iff]; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem ae_le_set_inter {s' t' : Set α} (h : s ≤ᵐ[μ] t) (h' : s' ≤ᵐ[μ] t') :
    (s ∩ s' : Set α) ≤ᵐ[μ] (t ∩ t' : Set α) :=
  h.inter h'


theorem ae_le_set_union {s' t' : Set α} (h : s ≤ᵐ[μ] t) (h' : s' ≤ᵐ[μ] t') :
    (s ∪ s' : Set α) ≤ᵐ[μ] (t ∪ t' : Set α) :=
  h.union h'


theorem union_ae_eq_right : (s ∪ t : Set α) =ᵐ[μ] t ↔ μ (s \ t) = 0 := by
  simp [eventuallyLE_antisymm_iff, ae_le_set, union_diff_right,
    diff_eq_empty.2 Set.subset_union_right]


theorem diff_ae_eq_self : (s \ t : Set α) =ᵐ[μ] s ↔ μ (s ∩ t) = 0 := by
  simp [eventuallyLE_antisymm_iff, ae_le_set, diff_diff_right, diff_diff,
    diff_eq_empty.2 Set.subset_union_right]


theorem diff_null_ae_eq_self (ht : μ t = 0) : (s \ t : Set α) =ᵐ[μ] s :=
  diff_ae_eq_self.mpr (measure_mono_null inter_subset_right ht)


theorem ae_eq_set {s t : Set α} : s =ᵐ[μ] t ↔ μ (s \ t) = 0 ∧ μ (t \ s) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq s t) (And (Eq (μ (SDiff.sdiff s t)) 0 …
  -/
  simp [eventuallyLE_antisymm_iff, ae_le_set]
  /-
    🎉 no goals
  -/


open scoped symmDiff in
@[simp]
theorem measure_symmDiff_eq_zero_iff {s t : Set α} : μ (s ∆ t) = 0 ↔ s =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ Iff (Eq (μ (symmDiff s t)) 0) ((MeasureTheory.ae μ).EventuallyEq s t)
  -/
  simp [ae_eq_set, symmDiff_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem ae_eq_set_compl_compl {s t : Set α} : sᶜ =ᵐ[μ] tᶜ ↔ s =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (HasCompl.compl s) (HasCompl.compl t) …
  -/
  simp only [← measure_symmDiff_eq_zero_iff, compl_symmDiff_compl]
  /-
    🎉 no goals
  -/


theorem ae_eq_set_compl {s t : Set α} : sᶜ =ᵐ[μ] t ↔ s =ᵐ[μ] tᶜ := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (HasCompl.compl s) t) ((MeasureTheory …
  -/
  rw [← ae_eq_set_compl_compl, compl_compl]
  /-
    🎉 no goals
  -/


theorem ae_eq_set_inter {s' t' : Set α} (h : s =ᵐ[μ] t) (h' : s' =ᵐ[μ] t') :
    (s ∩ s' : Set α) =ᵐ[μ] (t ∩ t' : Set α) :=
  h.inter h'


theorem ae_eq_set_union {s' t' : Set α} (h : s =ᵐ[μ] t) (h' : s' =ᵐ[μ] t') :
    (s ∪ s' : Set α) =ᵐ[μ] (t ∪ t' : Set α) :=
  h.union h'


theorem union_ae_eq_univ_of_ae_eq_univ_left (h : s =ᵐ[μ] univ) : (s ∪ t : Set α) =ᵐ[μ] univ :=
                                                 /-
                                                   α : Type u_1
                                                   F : Type u_3
                                                   inst✝¹ : FunLike F (Set α) ENNReal
                                                   inst✝ : MeasureTheory.OuterMeasureClass F α
                                                   μ : F
                                                   s t : Set α
                                                   h : (MeasureTheory.ae μ).EventuallyEq s Set.univ
                                                   ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union Set.univ t) Set.univ
                                                 -/
  (ae_eq_set_union h (ae_eq_refl t)).trans <| by rw [univ_union]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem union_ae_eq_univ_of_ae_eq_univ_right (h : t =ᵐ[μ] univ) : (s ∪ t : Set α) =ᵐ[μ] univ := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t Set.univ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union s t) Set.univ
  -/
  convert ae_eq_set_union (ae_eq_refl s) h
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t Set.univ
    ⊢ Eq Set.univ (Union.union s Set.univ)
  -/
  rw [union_univ]
  /-
    🎉 no goals
  -/


theorem union_ae_eq_right_of_ae_eq_empty (h : s =ᵐ[μ] (∅ : Set α)) : (s ∪ t : Set α) =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union s t) t
  -/
  convert ae_eq_set_union h (ae_eq_refl t)
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
    ⊢ Eq t (Union.union EmptyCollection.emptyCollection t)
  -/
  rw [empty_union]
  /-
    🎉 no goals
  -/


theorem union_ae_eq_left_of_ae_eq_empty (h : t =ᵐ[μ] (∅ : Set α)) : (s ∪ t : Set α) =ᵐ[μ] s := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t EmptyCollection.emptyCollection
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union s t) s
  -/
  convert ae_eq_set_union (ae_eq_refl s) h
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t EmptyCollection.emptyCollection
    ⊢ Eq s (Union.union s EmptyCollection.emptyCollection)
  -/
  rw [union_empty]
  /-
    🎉 no goals
  -/


theorem inter_ae_eq_right_of_ae_eq_univ (h : s =ᵐ[μ] univ) : (s ∩ t : Set α) =ᵐ[μ] t := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s Set.univ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter s t) t
  -/
  convert ae_eq_set_inter h (ae_eq_refl t)
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s Set.univ
    ⊢ Eq t (Inter.inter Set.univ t)
  -/
  rw [univ_inter]
  /-
    🎉 no goals
  -/


theorem inter_ae_eq_left_of_ae_eq_univ (h : t =ᵐ[μ] univ) : (s ∩ t : Set α) =ᵐ[μ] s := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t Set.univ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter s t) s
  -/
  convert ae_eq_set_inter (ae_eq_refl s) h
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t Set.univ
    ⊢ Eq s (Inter.inter s Set.univ)
  -/
  rw [inter_univ]
  /-
    🎉 no goals
  -/


theorem inter_ae_eq_empty_of_ae_eq_empty_left (h : s =ᵐ[μ] (∅ : Set α)) :
    (s ∩ t : Set α) =ᵐ[μ] (∅ : Set α) := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter s t) EmptyCollection.emptyCol …
  -/
  convert ae_eq_set_inter h (ae_eq_refl t)
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection
    ⊢ Eq EmptyCollection.emptyCollection (Inter.inter EmptyCollection.emptyCollect …
  -/
  rw [empty_inter]
  /-
    🎉 no goals
  -/


theorem inter_ae_eq_empty_of_ae_eq_empty_right (h : t =ᵐ[μ] (∅ : Set α)) :
    (s ∩ t : Set α) =ᵐ[μ] (∅ : Set α) := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t EmptyCollection.emptyCollection
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Inter.inter s t) EmptyCollection.emptyCol …
  -/
  convert ae_eq_set_inter (ae_eq_refl s) h
  /-
    case h.e'_5
    α : Type u_1
    F : Type u_3
    inst✝¹ : FunLike F (Set α) ENNReal
    inst✝ : MeasureTheory.OuterMeasureClass F α
    μ : F
    s t : Set α
    h : (MeasureTheory.ae μ).EventuallyEq t EmptyCollection.emptyCollection
    ⊢ Eq EmptyCollection.emptyCollection (Inter.inter s EmptyCollection.emptyColle …
  -/
  rw [inter_empty]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem _root_.Set.mulIndicator_ae_eq_one {M : Type*} [One M] {f : α → M} {s : Set α} :
    s.mulIndicator f =ᵐ[μ] 1 ↔ μ (s ∩ f.mulSupport) = 0 := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝² : FunLike F (Set α) ENNReal
    inst✝¹ : MeasureTheory.OuterMeasureClass F α
    μ : F
    M : Type u_4
    inst✝ : One M
    f : α → M
    s : Set α
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyEq (s.mulIndicator f) 1) (Eq (μ (Inter.i …
  -/
  simp [EventuallyEq, eventually_iff, ae, compl_setOf]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- If `s ⊆ t` modulo a set of measure `0`, then `μ s ≤ μ t`. -/
@[mono]
theorem measure_mono_ae (H : s ≤ᵐ[μ] t) : μ s ≤ μ t :=
  calc
    μ s ≤ μ (s ∪ t) := measure_mono subset_union_left
                            /-
                              α : Type u_1
                              F : Type u_3
                              inst✝¹ : FunLike F (Set α) ENNReal
                              inst✝ : MeasureTheory.OuterMeasureClass F α
                              μ : F
                              s t : Set α
                              H : (MeasureTheory.ae μ).EventuallyLE s t
                              ⊢ Eq (μ (Union.union s t)) (μ (Union.union t (SDiff.sdiff s t)))
                            -/
    _ = μ (t ∪ s \ t) := by rw [union_diff_self, Set.union_comm]
                            /-
                              🎉 no goals
                            -/
    _ ≤ μ t + μ (s \ t) := measure_union_le _ _
                  /-
                    α : Type u_1
                    F : Type u_3
                    inst✝¹ : FunLike F (Set α) ENNReal
                    inst✝ : MeasureTheory.OuterMeasureClass F α
                    μ : F
                    s t : Set α
                    H : (MeasureTheory.ae μ).EventuallyLE s t
                    ⊢ Eq (HAdd.hAdd (μ t) (μ (SDiff.sdiff s t))) (μ t)
                  -/
    _ = μ t := by rw [ae_le_set.1 H, add_zero]
                  /-
                    🎉 no goals
                  -/


alias _root_.Filter.EventuallyLE.measure_le := measure_mono_ae


/-- If two sets are equal modulo a set of measure zero, then `μ s = μ t`. -/
theorem measure_congr (H : s =ᵐ[μ] t) : μ s = μ t :=
  le_antisymm H.le.measure_le H.symm.le.measure_le


alias _root_.Filter.EventuallyEq.measure_eq := measure_congr


theorem measure_mono_null_ae (H : s ≤ᵐ[μ] t) (ht : μ t = 0) : μ s = 0 :=
  nonpos_iff_eq_zero.1 <| ht ▸ H.measure_le


