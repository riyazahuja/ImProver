/-- **Radon's theorem on convex sets**.

Any family `f` of affine dependent vectors contains a set `I` with the property that convex hulls of
`I` and `Iᶜ` intersect nontrivially. -/
theorem radon_partition {f : ι → E} (h : ¬ AffineIndependent 𝕜 f) :
    ∃ I, (convexHull 𝕜 (f '' I) ∩ convexHull 𝕜 (f '' Iᶜ)).Nonempty := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    h : Not (AffineIndependent 𝕜 f)
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  rw [affineIndependent_iff] at h
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    h : Not (∀ (s : Finset ι) (w : ι → 𝕜), Eq (s.sum w) 0 → Eq (s.sum fun e => HSM …
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  push_neg at h
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    h : Exists fun s => Exists fun w => And (Eq (s.sum w) 0) (And (Eq (s.sum fun e …
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  obtain ⟨s, w, h_wsum, h_vsum, nonzero_w_index, h1, h2⟩ := h
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    s : Finset ι
    w : ι → 𝕜
    h_wsum : Eq (s.sum w) 0
    h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
    nonzero_w_index : ι
    h1 : Membership.mem s nonzero_w_index
    h2 : Ne (w nonzero_w_index) 0
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  let I : Finset ι := {i ∈ s | 0 ≤ w i}
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    s : Finset ι
    w : ι → 𝕜
    h_wsum : Eq (s.sum w) 0
    h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
    nonzero_w_index : ι
    h1 : Membership.mem s nonzero_w_index
    h2 : Ne (w nonzero_w_index) 0
    I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  let J : Finset ι := {i ∈ s | w i < 0}
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    s : Finset ι
    w : ι → 𝕜
    h_wsum : Eq (s.sum w) 0
    h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
    nonzero_w_index : ι
    h1 : Membership.mem s nonzero_w_index
    h2 : Ne (w nonzero_w_index) 0
    I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
    J : Finset ι := Finset.filter (fun i => LT.lt (w i) 0) s
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  let p : E := centerMass I w f -- point of intersection
  have hJI : ∑ j ∈ J, w j + ∑ i ∈ I, w i = 0 := by
    simpa only [h_wsum, not_lt] using sum_filter_add_sum_filter_not s (fun i ↦ w i < 0) w
  have hI : 0 < ∑ i ∈ I, w i := by
    rcases exists_pos_of_sum_zero_of_exists_nonzero _ h_wsum ⟨nonzero_w_index, h1, h2⟩
      with ⟨pos_w_index, h1', h2'⟩
    exact sum_pos' (fun _i hi ↦ (mem_filter.1 hi).2)
      ⟨pos_w_index, by simp only [I, mem_filter, h1', h2'.le, and_self, h2']⟩
  have hp : centerMass J w f = p := centerMass_of_sum_add_sum_eq_zero hJI <| by
    simpa only [← h_vsum, not_lt] using sum_filter_add_sum_filter_not s (fun i ↦ w i < 0) _
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    s : Finset ι
    w : ι → 𝕜
    h_wsum : Eq (s.sum w) 0
    h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
    nonzero_w_index : ι
    h1 : Membership.mem s nonzero_w_index
    h2 : Ne (w nonzero_w_index) 0
    I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
    J : Finset ι := Finset.filter (fun i => LT.lt (w i) 0) s
    p : E := I.centerMass w f
    hJI : Eq (HAdd.hAdd (J.sum fun j => w j) (I.sum fun i => w i)) 0
    hI : LT.lt 0 (I.sum fun i => w i)
    hp : Eq (J.centerMass w f) p
    ⊢ Exists fun I => (Inter.inter ((convexHull 𝕜) (Set.image f I)) ((convexHull 𝕜 …
  -/
  refine ⟨I, p, ?_, ?_⟩
  · exact centerMass_mem_convexHull _ (fun _i hi ↦ (mem_filter.mp hi).2) hI
      (fun _i hi ↦ mem_image_of_mem _ hi)
  /-
    case intro.intro.intro.intro.intro.intro.refine_2
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : ι → E
    s : Finset ι
    w : ι → 𝕜
    h_wsum : Eq (s.sum w) 0
    h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
    nonzero_w_index : ι
    h1 : Membership.mem s nonzero_w_index
    h2 : Ne (w nonzero_w_index) 0
    I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
    J : Finset ι := Finset.filter (fun i => LT.lt (w i) 0) s
    p : E := I.centerMass w f
    hJI : Eq (HAdd.hAdd (J.sum fun j => w j) (I.sum fun i => w i)) 0
    hI : LT.lt 0 (I.sum fun i => w i)
    hp : Eq (J.centerMass w f) p
    ⊢ Membership.mem ((convexHull 𝕜) (Set.image f (HasCompl.compl ↑I))) p
  -/
  rw [← hp]
  refine centerMass_mem_convexHull_of_nonpos _ (fun _ hi ↦ (mem_filter.mp hi).2.le) ?_
    (fun _i hi ↦ mem_image_of_mem _ fun hi' ↦ ?_)
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.refine_1
      ι : Type u_1
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : ι → E
      s : Finset ι
      w : ι → 𝕜
      h_wsum : Eq (s.sum w) 0
      h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
      nonzero_w_index : ι
      h1 : Membership.mem s nonzero_w_index
      h2 : Ne (w nonzero_w_index) 0
      I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
      J : Finset ι := Finset.filter (fun i => LT.lt (w i) 0) s
      p : E := I.centerMass w f
      hJI : Eq (HAdd.hAdd (J.sum fun j => w j) (I.sum fun i => w i)) 0
      hI : LT.lt 0 (I.sum fun i => w i)
      hp : Eq (J.centerMass w f) p
      ⊢ LT.lt (J.sum fun i => w i) 0
    -/
  · linarith only [hI, hJI]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2.refine_2
      ι : Type u_1
      𝕜 : Type u_2
      E : Type u_3
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : ι → E
      s : Finset ι
      w : ι → 𝕜
      h_wsum : Eq (s.sum w) 0
      h_vsum : Eq (s.sum fun e => HSMul.hSMul (w e) (f e)) 0
      nonzero_w_index : ι
      h1 : Membership.mem s nonzero_w_index
      h2 : Ne (w nonzero_w_index) 0
      I : Finset ι := Finset.filter (fun i => LE.le 0 (w i)) s
      J : Finset ι := Finset.filter (fun i => LT.lt (w i) 0) s
      p : E := I.centerMass w f
      hJI : Eq (HAdd.hAdd (J.sum fun j => w j) (I.sum fun i => w i)) 0
      hI : LT.lt 0 (I.sum fun i => w i)
      hp : Eq (J.centerMass w f) p
      _i : ι
      hi : Membership.mem J _i
      hi' : Membership.mem (↑I) _i
      ⊢ False
    -/
  · exact (mem_filter.mp hi').2.not_lt (mem_filter.mp hi).2
    /-
      🎉 no goals
    -/


/-- Corner case for `helly_theorem'`. -/
private lemma helly_theorem_corner {F : ι → Set E} {s : Finset ι}
    (h_card_small : #s ≤ finrank 𝕜 E + 1)
    (h_inter : ∀ I ⊆ s, #I ≤ finrank 𝕜 E + 1 → (⋂ i ∈ I, F i).Nonempty) :
                                             /-
                                               ι : Type u_1
                                               𝕜 : Type u_2
                                               E : Type u_3
                                               inst✝² : LinearOrderedField 𝕜
                                               inst✝¹ : AddCommGroup E
                                               inst✝ : Module 𝕜 E
                                               F : ι → Set E
                                               s : Finset ι
                                               h_card_small : LE.le s.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
                                               h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → LE.le I.card (HAdd.hAdd (Mo …
                                               ⊢ HasSubset.Subset s s
                                             -/
    (⋂ i ∈ s, F i).Nonempty := h_inter s (by simp) h_card_small
                                             /-
                                               🎉 no goals
                                             -/


/-- **Helly's theorem** for finite families of convex sets.

If `F` is a finite family of convex sets in a vector space of finite dimension `d`, and any
`k ≤ d + 1` sets of `F` intersect nontrivially, then all sets of `F` intersect nontrivially. -/
theorem helly_theorem' {F : ι → Set E} {s : Finset ι}
    (h_convex : ∀ i ∈ s, Convex 𝕜 (F i))
    (h_inter : ∀ I ⊆ s, #I ≤ finrank 𝕜 E + 1 → (⋂ i ∈ I, F i).Nonempty) :
    (⋂ i ∈ s, F i).Nonempty := by
  classical
  obtain h_card | h_card := lt_or_le #s (finrank 𝕜 E + 1)
  · exact helly_theorem_corner (le_of_lt h_card) h_inter
  generalize hn : #s = n
  rw [hn] at h_card
  induction' n, h_card using Nat.le_induction with k h_card hk generalizing ι
  · exact helly_theorem_corner (le_of_eq hn) h_inter
  /- Construct a family of vectors indexed by `ι` such that the vector corresponding to `i : ι`
  is an arbitrary element of the intersection of all `F j` except `F i`. -/
  let a (i : s) : E := Set.Nonempty.some (s := ⋂ j ∈ s.erase i, F j) <| by
    apply hk (s := s.erase i)
    · exact fun i hi ↦ h_convex i (mem_of_mem_erase hi)
    · intro J hJ_ss hJ_card
      exact h_inter J (subset_trans hJ_ss (erase_subset i.val s)) hJ_card
    · simp only [coe_mem, card_erase_of_mem]; omega
  /- This family of vectors is not affine independent because the number of them exceeds the
  dimension of the space. -/
  have h_ind : ¬AffineIndependent 𝕜 a := by
    rw [← finrank_vectorSpan_le_iff_not_affineIndependent 𝕜 a (n := (k - 1))]
    · exact (Submodule.finrank_le (vectorSpan 𝕜 (range a))).trans (Nat.le_pred_of_lt h_card)
    · simp only [card_coe]; omega
  /- Use `radon_partition` to conclude there is a subset `I` of `s` and a point `p : E` which
  lies in the convex hull of either `a '' I` or `a '' Iᶜ`. We claim that `p ∈ ⋂ i ∈ s, F i`. -/
  obtain ⟨I, p, hp_I, hp_Ic⟩ := radon_partition h_ind
  use p
  apply mem_biInter
  intro i hi
  let i : s := ⟨i, hi⟩
  /- It suffices to show that for any subcollection `J` of `s` containing `i`, the convex
  hull of `a '' (s \ J)` is contained in `F i`. -/
  suffices ∀ J : Set s, (i ∈ J) → (convexHull 𝕜) (a '' Jᶜ) ⊆ F i by
    by_cases h : i ∈ I
    · exact this I h hp_Ic
    · apply this Iᶜ h; rwa [compl_compl]
  /- Given any subcollection `J` of `ι` containing `i`, because `F i` is convex, we need only
  show that `a j ∈ F i` for each `j ∈ s \ J`. -/
  intro J hi
  rw [convexHull_subset_iff (h_convex i.1 i.2)]
  rintro v ⟨j, hj, hj_v⟩
  rw [← hj_v]
  /- Since `j ∈ Jᶜ` and `i ∈ J`, we conclude that `i ≠ j`, and hence by the definition of `a`:
  `a j ∈ ⋂ F '' (Set.univ \ {j}) ⊆ F i`. -/
  apply mem_of_subset_of_mem (s₁ := ⋂ k ∈ (s.erase j), F k)
  · apply biInter_subset_of_mem
    simp only [erase_val]
    suffices h : i.val ∈ s.erase j by assumption
    simp only [mem_erase]
    constructor
    · exact fun h' ↦ hj ((show i = j from SetCoe.ext h') ▸ hi)
    · assumption
  · apply Nonempty.some_mem


/-- **Helly's theorem** for finite families of convex sets in its classical form.

If `F` is a family of `n` convex sets in a vector space of finite dimension `d`, with `n ≥ d + 1`,
and any `d + 1` sets of `F` intersect nontrivially, then all sets of `F` intersect nontrivially. -/
theorem helly_theorem {F : ι → Set E} {s : Finset ι}
    (h_card : finrank 𝕜 E + 1 ≤ #s)
    (h_convex : ∀ i ∈ s, Convex 𝕜 (F i))
    (h_inter : ∀ I ⊆ s, #I = finrank 𝕜 E + 1 → (⋂ i ∈ I, F i).Nonempty) :
    (⋂ i ∈ s, F i).Nonempty := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : ι → Set E
    s : Finset ι
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) s.card
    h_convex : ∀ (i : ι), Membership.mem s i → Convex 𝕜 (F i)
    h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → Eq I.card (HAdd.hAdd (Modul …
    ⊢ (Set.iInter fun i => Set.iInter fun h => F i).Nonempty
  -/
  apply helly_theorem' h_convex
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : ι → Set E
    s : Finset ι
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) s.card
    h_convex : ∀ (i : ι), Membership.mem s i → Convex 𝕜 (F i)
    h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → Eq I.card (HAdd.hAdd (Modul …
    ⊢ ∀ (I : Finset ι), HasSubset.Subset I s → LE.le I.card (HAdd.hAdd (Module.fin …
  -/
  intro I hI_ss hI_card
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : ι → Set E
    s : Finset ι
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) s.card
    h_convex : ∀ (i : ι), Membership.mem s i → Convex 𝕜 (F i)
    h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → Eq I.card (HAdd.hAdd (Modul …
    I : Finset ι
    hI_ss : HasSubset.Subset I s
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (Set.iInter fun i => Set.iInter fun h => F i).Nonempty
  -/
  obtain ⟨J, hI_ss_J, hJ_ss, hJ_card⟩ := exists_subsuperset_card_eq hI_ss hI_card h_card
  /-
    case intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : ι → Set E
    s : Finset ι
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) s.card
    h_convex : ∀ (i : ι), Membership.mem s i → Convex 𝕜 (F i)
    h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → Eq I.card (HAdd.hAdd (Modul …
    I : Finset ι
    hI_ss : HasSubset.Subset I s
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset ι
    hI_ss_J : HasSubset.Subset I J
    hJ_ss : HasSubset.Subset J s
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (Set.iInter fun i => Set.iInter fun h => F i).Nonempty
  -/
  apply Set.Nonempty.mono <| biInter_mono hI_ss_J (fun _ _ ↦ Set.Subset.rfl)
  /-
    case intro.intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : ι → Set E
    s : Finset ι
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) s.card
    h_convex : ∀ (i : ι), Membership.mem s i → Convex 𝕜 (F i)
    h_inter : ∀ (I : Finset ι), HasSubset.Subset I s → Eq I.card (HAdd.hAdd (Modul …
    I : Finset ι
    hI_ss : HasSubset.Subset I s
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset ι
    hI_ss_J : HasSubset.Subset I J
    hJ_ss : HasSubset.Subset J s
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (Set.iInter fun x => Set.iInter fun h => F x).Nonempty
  -/
  exact h_inter J hJ_ss hJ_card
  /-
    🎉 no goals
  -/


/-- **Helly's theorem** for finite sets of convex sets.

If `F` is a finite set of convex sets in a vector space of finite dimension `d`, and any `k ≤ d + 1`
sets from `F` intersect nontrivially, then all sets from `F` intersect nontrivially. -/
theorem helly_theorem_set' {F : Finset (Set E)}
    (h_convex : ∀ X ∈ F, Convex 𝕜 X)
    (h_inter : ∀ G : Finset (Set E), G ⊆ F → #G ≤ finrank 𝕜 E + 1 → (⋂₀ G : Set E).Nonempty) :
    (⋂₀ (F : Set (Set E))).Nonempty := by
  classical -- for DecidableEq, required for the family version
  rw [show ⋂₀ F = ⋂ X ∈ F, (X : Set E) by ext; simp]
  apply helly_theorem' h_convex
  intro G hG_ss hG_card
  rw [show ⋂ X ∈ G, X = ⋂₀ G by ext; simp]
  exact h_inter G hG_ss hG_card


/-- **Helly's theorem** for finite sets of convex sets in its classical form.

If `F` is a finite set of convex sets in a vector space of finite dimension `d`, with `n ≥ d + 1`,
and any `d + 1` sets from `F` intersect nontrivially,
then all sets from `F` intersect nontrivially. -/
theorem helly_theorem_set {F : Finset (Set E)}
    (h_card : finrank 𝕜 E + 1 ≤ #F)
    (h_convex : ∀ X ∈ F, Convex 𝕜 X)
    (h_inter : ∀ G : Finset (Set E), G ⊆ F → #G = finrank 𝕜 E + 1 → (⋂₀ G : Set E).Nonempty) :
    (⋂₀ (F : Set (Set E))).Nonempty := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    ⊢ (↑F).sInter.Nonempty
  -/
  apply helly_theorem_set' h_convex
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    ⊢ ∀ (G : Finset (Set E)), HasSubset.Subset G F → LE.le G.card (HAdd.hAdd (Modu …
  -/
  intro I hI_ss hI_card
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset I F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (↑I).sInter.Nonempty
  -/
  obtain ⟨J, _, hJ_ss, hJ_card⟩ := exists_subsuperset_card_eq hI_ss hI_card h_card
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset I F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset (Set E)
    left✝ : HasSubset.Subset I J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (↑I).sInter.Nonempty
  -/
  have : ⋂₀ (J : Set (Set E)) ⊆ ⋂₀ I := sInter_mono (by simpa [hI_ss])
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset I F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset (Set E)
    left✝ : HasSubset.Subset I J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    this : HasSubset.Subset (↑J).sInter (↑I).sInter
    ⊢ (↑I).sInter.Nonempty
  -/
  apply Set.Nonempty.mono this
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    F : Finset (Set E)
    h_card : LE.le (HAdd.hAdd (Module.finrank 𝕜 E) 1) F.card
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset G F → Eq G.card (HAdd.hAdd  …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset I F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset (Set E)
    left✝ : HasSubset.Subset I J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    this : HasSubset.Subset (↑J).sInter (↑I).sInter
    ⊢ (↑J).sInter.Nonempty
  -/
  exact h_inter J hJ_ss (by omega)
  /-
    🎉 no goals
  -/


/-- **Helly's theorem** for families of compact convex sets.

If `F` is a family of compact convex sets in a vector space of finite dimension `d`, and any
`k ≤ d + 1` sets of `F` intersect nontrivially, then all sets of `F` intersect nontrivially. -/
theorem helly_theorem_compact' [TopologicalSpace E] [T2Space E] {F : ι → Set E}
    (h_convex : ∀ i, Convex 𝕜 (F i)) (h_compact : ∀ i, IsCompact (F i))
    (h_inter : ∀ I : Finset ι, #I ≤ finrank 𝕜 E + 1 → (⋂ i ∈ I, F i).Nonempty) :
    (⋂ i, F i).Nonempty := by
  classical
  /- If `ι` is empty the statement is trivial. -/
  cases' isEmpty_or_nonempty ι with _ h_nonempty
  · simp only [iInter_of_empty, Set.univ_nonempty]
  /- By the finite version of theorem, every finite subfamily has an intersection. -/
  have h_fin (I : Finset ι) : (⋂ i ∈ I, F i).Nonempty := by
    apply helly_theorem' (s := I) (𝕜 := 𝕜) (by simp [h_convex])
    exact fun J _ hJ_card ↦ h_inter J hJ_card
  /- The following is a clumsy proof that family of compact sets with the finite intersection
  property has a nonempty intersection. -/
  have i0 : ι := Nonempty.some h_nonempty
  rw [show ⋂ i, F i = (F i0) ∩ ⋂ i, F i by simp [iInter_subset]]
  apply IsCompact.inter_iInter_nonempty
  · exact h_compact i0
  · intro i
    exact (h_compact i).isClosed
  · intro I
    simpa using h_fin ({i0} ∪ I)


/-- **Helly's theorem** for families of compact convex sets in its classical form.

If `F` is a (possibly infinite) family of more than `d + 1` compact convex sets in a vector space of
finite dimension `d`, and any `d + 1` sets of `F` intersect nontrivially,
then all sets of `F` intersect nontrivially. -/
theorem helly_theorem_compact [TopologicalSpace E] [T2Space E] {F : ι → Set E}
    (h_card : finrank 𝕜 E + 1 ≤ ENat.card ι)
    (h_convex : ∀ i, Convex 𝕜 (F i)) (h_compact : ∀ i, IsCompact (F i))
    (h_inter : ∀ I : Finset ι, #I = finrank 𝕜 E + 1 → (⋂ i ∈ I, F i).Nonempty) :
    (⋂ i, F i).Nonempty := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : ι → Set E
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) (ENat.card ι)
    h_convex : ∀ (i : ι), Convex 𝕜 (F i)
    h_compact : ∀ (i : ι), IsCompact (F i)
    h_inter : ∀ (I : Finset ι), Eq I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Se …
    ⊢ (Set.iInter fun i => F i).Nonempty
  -/
  apply helly_theorem_compact' h_convex h_compact
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : ι → Set E
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) (ENat.card ι)
    h_convex : ∀ (i : ι), Convex 𝕜 (F i)
    h_compact : ∀ (i : ι), IsCompact (F i)
    h_inter : ∀ (I : Finset ι), Eq I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Se …
    ⊢ ∀ (I : Finset ι), LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Set.iIn …
  -/
  intro I hI_card
  have hJ : ∃ J : Finset ι, I ⊆ J ∧ #J = finrank 𝕜 E + 1 := by
    by_cases h : Infinite ι
    · exact Infinite.exists_superset_card_eq _ _ hI_card
    · have : Finite ι := Finite.of_not_infinite h
      have : Fintype ι := Fintype.ofFinite ι
      apply exists_superset_card_eq hI_card
      simp only [ENat.card_eq_coe_fintype_card] at h_card
      rwa [← Nat.cast_one, ← Nat.cast_add, Nat.cast_le] at h_card
  /-
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : ι → Set E
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) (ENat.card ι)
    h_convex : ∀ (i : ι), Convex 𝕜 (F i)
    h_compact : ∀ (i : ι), IsCompact (F i)
    h_inter : ∀ (I : Finset ι), Eq I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Se …
    I : Finset ι
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    hJ : Exists fun J => And (HasSubset.Subset I J) (Eq J.card (HAdd.hAdd (Module. …
    ⊢ (Set.iInter fun i => Set.iInter fun h => F i).Nonempty
  -/
  obtain ⟨J, hJ_ss, hJ_card⟩ := hJ
  /-
    case intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : ι → Set E
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) (ENat.card ι)
    h_convex : ∀ (i : ι), Convex 𝕜 (F i)
    h_compact : ∀ (i : ι), IsCompact (F i)
    h_inter : ∀ (I : Finset ι), Eq I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Se …
    I : Finset ι
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset ι
    hJ_ss : HasSubset.Subset I J
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (Set.iInter fun i => Set.iInter fun h => F i).Nonempty
  -/
  apply Set.Nonempty.mono <| biInter_mono hJ_ss (by intro _ _; rfl)
  /-
    case intro.intro
    ι : Type u_1
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : ι → Set E
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) (ENat.card ι)
    h_convex : ∀ (i : ι), Convex 𝕜 (F i)
    h_compact : ∀ (i : ι), IsCompact (F i)
    h_inter : ∀ (I : Finset ι), Eq I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1) → (Se …
    I : Finset ι
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Finset ι
    hJ_ss : HasSubset.Subset I J
    hJ_card : Eq J.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    ⊢ (Set.iInter fun x => Set.iInter fun h => F x).Nonempty
  -/
  exact h_inter J hJ_card
  /-
    🎉 no goals
  -/


/-- **Helly's theorem** for sets of compact convex sets.

If `F` is a set of compact convex sets in a vector space of finite dimension `d`, and any
`k ≤ d + 1` sets from `F` intersect nontrivially, then all sets from `F` intersect nontrivially. -/
theorem helly_theorem_set_compact' [TopologicalSpace E] [T2Space E] {F : Set (Set E)}
    (h_convex : ∀ X ∈ F, Convex 𝕜 X) (h_compact : ∀ X ∈ F, IsCompact X)
    (h_inter : ∀ G : Finset (Set E), (G : Set (Set E)) ⊆ F → #G ≤ finrank 𝕜 E + 1 →
    (⋂₀ G : Set E).Nonempty) :
    (⋂₀ (F : Set (Set E))).Nonempty := by
  classical -- for DecidableEq, required for the family version
  rw [show ⋂₀ F = ⋂ X : F, (X : Set E) by ext; simp]
  refine helly_theorem_compact' (F := fun x : F ↦ x.val)
    (fun X ↦ h_convex X (by simp)) (fun X ↦ h_compact X (by simp)) ?_
  intro G _
  let G' : Finset (Set E) := image Subtype.val G
  rw [show ⋂ i ∈ G, ↑i = ⋂₀ (G' : Set (Set E)) by simp [G']]
  apply h_inter G'
  · simp [G']
  · apply le_trans card_image_le
    assumption


/-- **Helly's theorem** for sets of compact convex sets in its classical version.

If `F` is a (possibly infinite) set of more than `d + 1` compact convex sets in a vector space of
finite dimension `d`, and any `d + 1` sets from `F` intersect nontrivially,
then all sets from `F` intersect nontrivially. -/
theorem helly_theorem_set_compact [TopologicalSpace E] [T2Space E] {F : Set (Set E)}
    (h_card : finrank 𝕜 E + 1 ≤ F.encard)
    (h_convex : ∀ X ∈ F, Convex 𝕜 X) (h_compact : ∀ X ∈ F, IsCompact X)
    (h_inter : ∀ G : Finset (Set E), (G : Set (Set E)) ⊆ F → #G = finrank 𝕜 E + 1 →
    (⋂₀ G : Set E).Nonempty) :
    (⋂₀ (F : Set (Set E))).Nonempty := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    ⊢ F.sInter.Nonempty
  -/
  apply helly_theorem_set_compact' h_convex h_compact
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    ⊢ ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → LE.le G.card (HAdd.hAdd (M …
  -/
  intro I hI_ss hI_card
  obtain ⟨J, _, hJ_ss, hJ_card⟩ := exists_superset_subset_encard_eq hI_ss (hkt := h_card)
    (by simpa only [encard_coe_eq_coe_finsetCard, ← ENat.coe_one, ← ENat.coe_add, Nat.cast_le])
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset (↑I) F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Set (Set E)
    left✝ : HasSubset.Subset (↑I) J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
    ⊢ (↑I).sInter.Nonempty
  -/
  apply Set.Nonempty.mono <| sInter_mono (by simpa [hI_ss])
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset (↑I) F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Set (Set E)
    left✝ : HasSubset.Subset (↑I) J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
    ⊢ J.sInter.Nonempty
  -/
  have hJ_fin : Fintype J := Finite.fintype <| finite_of_encard_eq_coe hJ_card
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset (↑I) F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Set (Set E)
    left✝ : HasSubset.Subset (↑I) J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
    hJ_fin : Fintype ↑J
    ⊢ J.sInter.Nonempty
  -/
  let J' := J.toFinset
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset (↑I) F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Set (Set E)
    left✝ : HasSubset.Subset (↑I) J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
    hJ_fin : Fintype ↑J
    J' : Finset (Set E) := J.toFinset
    ⊢ J.sInter.Nonempty
  -/
  rw [← coe_toFinset J]
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : FiniteDimensional 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : T2Space E
    F : Set (Set E)
    h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
    h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
    h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
    h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
    I : Finset (Set E)
    hI_ss : HasSubset.Subset (↑I) F
    hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    J : Set (Set E)
    left✝ : HasSubset.Subset (↑I) J
    hJ_ss : HasSubset.Subset J F
    hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
    hJ_fin : Fintype ↑J
    J' : Finset (Set E) := J.toFinset
    ⊢ (↑J.toFinset).sInter.Nonempty
  -/
  apply h_inter J'
    /-
      case intro.intro.intro.a
      𝕜 : Type u_2
      E : Type u_3
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : FiniteDimensional 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : T2Space E
      F : Set (Set E)
      h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
      h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
      h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
      h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
      I : Finset (Set E)
      hI_ss : HasSubset.Subset (↑I) F
      hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
      J : Set (Set E)
      left✝ : HasSubset.Subset (↑I) J
      hJ_ss : HasSubset.Subset J F
      hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
      hJ_fin : Fintype ↑J
      J' : Finset (Set E) := J.toFinset
      ⊢ HasSubset.Subset (↑J') F
    -/
  · simpa [J']
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.a
      𝕜 : Type u_2
      E : Type u_3
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : FiniteDimensional 𝕜 E
      inst✝¹ : TopologicalSpace E
      inst✝ : T2Space E
      F : Set (Set E)
      h_card : LE.le (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1) F.encard
      h_convex : ∀ (X : Set E), Membership.mem F X → Convex 𝕜 X
      h_compact : ∀ (X : Set E), Membership.mem F X → IsCompact X
      h_inter : ∀ (G : Finset (Set E)), HasSubset.Subset (↑G) F → Eq G.card (HAdd.hA …
      I : Finset (Set E)
      hI_ss : HasSubset.Subset (↑I) F
      hI_card : LE.le I.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
      J : Set (Set E)
      left✝ : HasSubset.Subset (↑I) J
      hJ_ss : HasSubset.Subset J F
      hJ_card : Eq J.encard (HAdd.hAdd (↑(Module.finrank 𝕜 E)) 1)
      hJ_fin : Fintype ↑J
      J' : Finset (Set E) := J.toFinset
      ⊢ Eq J'.card (HAdd.hAdd (Module.finrank 𝕜 E) 1)
    -/
  · rwa [encard_eq_coe_toFinset_card J, ← ENat.coe_one, ← ENat.coe_add, Nat.cast_inj] at hJ_card
    /-
      🎉 no goals
    -/


