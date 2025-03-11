lemma IsDynCoverOf.image (h : Semiconj φ S T) {F : Set X} {V : Set (Y × Y)} {n : ℕ} {s : Set X}
    (h' : IsDynCoverOf S F ((map φ φ) ⁻¹' V) n s) :
    IsDynCoverOf T (φ '' F) V n (φ '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    n : Nat
    s : Set X
    h' : Dynamics.IsDynCoverOf S F (Set.preimage (Prod.map φ φ) V) n s
    ⊢ Dynamics.IsDynCoverOf T (Set.image φ F) V n (Set.image φ s)
  -/
  simp only [IsDynCoverOf, image_subset_iff, preimage_iUnion₂, biUnion_image]
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    n : Nat
    s : Set X
    h' : Dynamics.IsDynCoverOf S F (Set.preimage (Prod.map φ φ) V) n s
    ⊢ HasSubset.Subset F (Set.iUnion fun i => Set.iUnion fun j => Set.preimage φ ( …
  -/
  refine h'.trans (iUnion₂_mono fun i _ ↦ subset_of_eq ?_)
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    n : Nat
    s : Set X
    h' : Dynamics.IsDynCoverOf S F (Set.preimage (Prod.map φ φ) V) n s
    i : X
    x✝ : Membership.mem s i
    ⊢ Eq (UniformSpace.ball i (Dynamics.dynEntourage S (Set.preimage (Prod.map φ φ …
  -/
  rw [← h.preimage_dynEntourage V n, ball_preimage]
  /-
    🎉 no goals
  -/


lemma IsDynCoverOf.preimage (h : Semiconj φ S T) {F : Set X} {V : Set (Y × Y)}
    (V_symm : SymmetricRel V) {n : ℕ} {t : Finset Y} (h' : IsDynCoverOf T (φ '' F) V n t) :
    ∃ s : Finset X, IsDynCoverOf S F ((map φ φ) ⁻¹' (V ○ V)) n s ∧ s.card ≤ t.card := by
  classical
  rcases isEmpty_or_nonempty X with _ | _
  · exact ⟨∅, eq_empty_of_isEmpty F ▸ ⟨isDynCoverOf_empty, Finset.card_empty ▸ zero_le t.card⟩⟩
  -- If `t` is a dynamical cover of `φ '' F`, then we want to choose one preimage by `φ` for each
  -- element of `t`. This is complicated by the fact that `t` may not be a subset of `φ '' F`,
  -- and may not even be in the range of `φ`. Hence, we first modify `t` to make it a subset
  -- of `φ '' F`. This requires taking larger entourages.
  rcases h'.nonempty_inter with ⟨s, s_cover, s_card, s_inter⟩
  choose! g gs_cover using fun (x : Y) (h : x ∈ s) ↦ nonempty_def.1 (s_inter x h)
  choose! f f_section using fun (y : Y) (a : y ∈ φ '' F) ↦ a
  refine ⟨s.image (f ∘ g), And.intro ?_ (Finset.card_image_le.trans s_card)⟩
  simp only [IsDynCoverOf, Finset.mem_coe, image_subset_iff, preimage_iUnion₂] at s_cover ⊢
  apply s_cover.trans
  rw [← h.preimage_dynEntourage (V ○ V) n, Finset.set_biUnion_finset_image]
  refine iUnion₂_mono fun i i_s ↦ ?_
  rw [comp_apply, ball_preimage, (f_section (g i) (gs_cover i i_s).2).2]
  refine preimage_mono fun x x_i ↦ mem_ball_dynEntourage_comp T n V_symm x (g i) ⟨i, ?_⟩
  replace gs_cover := (gs_cover i i_s).1
  rw [mem_ball_symmetry (V_symm.dynEntourage T n)] at x_i gs_cover
  exact ⟨x_i, gs_cover⟩


lemma le_coverMincard_image (h : Semiconj φ S T) (F : Set X) {V : Set (Y × Y)}
    (V_symm : SymmetricRel V) (n : ℕ) :
    coverMincard S F ((map φ φ) ⁻¹' (V ○ V)) n ≤ coverMincard T (φ '' F) V n := by
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    n : Nat
    ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
  -/
  rcases eq_top_or_lt_top (coverMincard T (φ '' F) V n) with h' | h'
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_symm : SymmetricRel V
      n : Nat
      h' : Eq (Dynamics.coverMincard T (Set.image φ F) V n) Top.top
      ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
    -/
  · exact h' ▸ le_top
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    n : Nat
    h' : LT.lt (Dynamics.coverMincard T (Set.image φ F) V n) Top.top
    ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
  -/
  rcases (coverMincard_finite_iff T (φ '' F) V n).1 h' with ⟨t, t_cover, t_card⟩
  /-
    case inr.intro.intro
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    n : Nat
    h' : LT.lt (Dynamics.coverMincard T (Set.image φ F) V n) Top.top
    t : Finset Y
    t_cover : Dynamics.IsDynCoverOf T (Set.image φ F) V n ↑t
    t_card : Eq (↑t.card) (Dynamics.coverMincard T (Set.image φ F) V n)
    ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
  -/
  rcases t_cover.preimage h V_symm with ⟨s, s_cover, s_card⟩
  /-
    case inr.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    n : Nat
    h' : LT.lt (Dynamics.coverMincard T (Set.image φ F) V n) Top.top
    t : Finset Y
    t_cover : Dynamics.IsDynCoverOf T (Set.image φ F) V n ↑t
    t_card : Eq (↑t.card) (Dynamics.coverMincard T (Set.image φ F) V n)
    s : Finset X
    s_cover : Dynamics.IsDynCoverOf S F (Set.preimage (Prod.map φ φ) (compRel V V) …
    s_card : LE.le s.card t.card
    ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
  -/
  rw [← t_card]
  /-
    case inr.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    n : Nat
    h' : LT.lt (Dynamics.coverMincard T (Set.image φ F) V n) Top.top
    t : Finset Y
    t_cover : Dynamics.IsDynCoverOf T (Set.image φ F) V n ↑t
    t_card : Eq (↑t.card) (Dynamics.coverMincard T (Set.image φ F) V n)
    s : Finset X
    s_cover : Dynamics.IsDynCoverOf S F (Set.preimage (Prod.map φ φ) (compRel V V) …
    s_card : LE.le s.card t.card
    ⊢ LE.le (Dynamics.coverMincard S F (Set.preimage (Prod.map φ φ) (compRel V V)) …
  -/
  exact s_cover.coverMincard_le_card.trans (WithTop.coe_le_coe.2 s_card)
  /-
    🎉 no goals
  -/


lemma coverMincard_image_le (h : Semiconj φ S T) (F : Set X) (V : Set (Y × Y)) (n : ℕ) :
    coverMincard T (φ '' F) V n ≤ coverMincard S F ((map φ φ) ⁻¹' V) n := by
  classical
  rcases eq_top_or_lt_top (coverMincard S F ((map φ φ) ⁻¹' V) n) with h' | h'
  · exact h' ▸ le_top
  rcases (coverMincard_finite_iff S F ((map φ φ) ⁻¹' V) n).1 h' with ⟨s, s_cover, s_card⟩
  rw [← s_card]
  have := s_cover.image h
  rw [← s.coe_image] at this
  exact this.coverMincard_le_card.trans (WithTop.coe_le_coe.2 s.card_image_le)


lemma le_coverEntropyEntourage_image (h : Semiconj φ S T) (F : Set X) {V : Set (Y × Y)}
    (V_symm : SymmetricRel V) :
    coverEntropyEntourage S F ((map φ φ) ⁻¹' (V ○ V)) ≤ coverEntropyEntourage T (φ '' F) V :=
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => HDi …
  -/
  /-
    🎉 no goals
  -/
  limsup_le_limsup (Eventually.of_forall fun n ↦ (monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
  /-
    🎉 no goals
  -/
    (log_monotone (ENat.toENNReal_mono (le_coverMincard_image h F V_symm n)))))


lemma le_coverEntropyInfEntourage_image (h : Semiconj φ S T) (F : Set X) {V : Set (Y × Y)}
    (V_symm : SymmetricRel V) :
    coverEntropyInfEntourage S F ((map φ φ) ⁻¹' (V ○ V)) ≤ coverEntropyInfEntourage T (φ '' F) V :=
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    V_symm : SymmetricRel V
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop fun n => HDiv. …
  -/
  /-
    🎉 no goals
  -/
  liminf_le_liminf (Eventually.of_forall fun n ↦ (monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
  /-
    🎉 no goals
  -/
    (log_monotone (ENat.toENNReal_mono (le_coverMincard_image h F V_symm n)))))


lemma coverEntropyEntourage_image_le (h : Semiconj φ S T) (F : Set X) (V : Set (Y × Y)) :
    coverEntropyEntourage T (φ '' F) V ≤ coverEntropyEntourage S F ((map φ φ) ⁻¹' V) :=
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => HDi …
  -/
  /-
    🎉 no goals
  -/
  limsup_le_limsup (Eventually.of_forall fun n ↦ (monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
  /-
    🎉 no goals
  -/
    (log_monotone (ENat.toENNReal_mono (coverMincard_image_le h F V n)))))


lemma coverEntropyInfEntourage_image_le (h : Semiconj φ S T) (F : Set X) (V : Set (Y × Y)) :
    coverEntropyInfEntourage T (φ '' F) V ≤ coverEntropyInfEntourage S F ((map φ φ) ⁻¹' V) :=
  /-
    X : Type u_1
    Y : Type u_2
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    V : Set (Prod Y Y)
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop fun n => HDiv. …
  -/
  /-
    🎉 no goals
  -/
  liminf_le_liminf (Eventually.of_forall fun n ↦ (monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
  /-
    🎉 no goals
  -/
    (log_monotone (ENat.toENNReal_mono (coverMincard_image_le h F V n)))))


/-- The entropy of `φ '' F` equals the entropy of `F` if `X` is endowed with the pullback by `φ`
  of the uniform structure of `Y`.-/
theorem coverEntropy_image_of_comap (u : UniformSpace Y) {S : X → X} {T : Y → Y} {φ : X → Y}
    (h : Semiconj φ S T) (F : Set X) :
    coverEntropy T (φ '' F) = @coverEntropy X (comap φ u) S F := by
  /-
    X : Type u_1
    Y : Type u_2
    u : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    ⊢ Eq (Dynamics.coverEntropy T (Set.image φ F)) (Dynamics.coverEntropy S F)
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      ⊢ LE.le (Dynamics.coverEntropy T (Set.image φ F)) (Dynamics.coverEntropy S F)
    -/
  · refine iSup₂_le fun V V_uni ↦ (coverEntropyEntourage_image_le h F V).trans ?_
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ LE.le (Dynamics.coverEntropyEntourage S F (Set.preimage (Prod.map φ φ) V)) ( …
    -/
    apply @coverEntropyEntourage_le_coverEntropy X (comap φ u) S F
    /-
      case a.h
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ Membership.mem (uniformity X) (Set.preimage (Prod.map φ φ) V)
    -/
    rw [uniformity_comap φ, mem_comap]
    /-
      case a.h
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ Exists fun t => And (Membership.mem (uniformity Y) t) (HasSubset.Subset (Set …
    -/
    exact ⟨V, V_uni, Subset.rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      ⊢ LE.le (Dynamics.coverEntropy S F) (Dynamics.coverEntropy T (Set.image φ F))
    -/
  · refine iSup₂_le fun U U_uni ↦ ?_
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.coverEntropyEntourage S F U) (Dynamics.coverEntropy T (Set.i …
    -/
    simp only [uniformity_comap φ, mem_comap] at U_uni
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      U_uni : Exists fun t => And (Membership.mem (uniformity Y) t) (HasSubset.Subse …
      ⊢ LE.le (Dynamics.coverEntropyEntourage S F U) (Dynamics.coverEntropy T (Set.i …
    -/
    rcases U_uni with ⟨V, V_uni, V_sub⟩
    /-
      case a.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      ⊢ LE.le (Dynamics.coverEntropyEntourage S F U) (Dynamics.coverEntropy T (Set.i …
    -/
    rcases comp_symm_mem_uniformity_sets V_uni with ⟨W, W_uni, W_symm, W_V⟩
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le (Dynamics.coverEntropyEntourage S F U) (Dynamics.coverEntropy T (Set.i …
    -/
    apply (coverEntropyEntourage_antitone S F ((preimage_mono W_V).trans V_sub)).trans
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le ((fun U => Dynamics.coverEntropyEntourage S F U) (Set.preimage (Prod.m …
    -/
    apply (le_coverEntropyEntourage_image h F W_symm).trans
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le (Dynamics.coverEntropyEntourage T (Set.image φ F) W) (Dynamics.coverEn …
    -/
    exact coverEntropyEntourage_le_coverEntropy T (φ '' F) W_uni
    /-
      🎉 no goals
    -/


/-- The entropy of `φ '' F` equals the entropy of `F` if `X` is endowed with the pullback by `φ`
  of the uniform structure of `Y`. This version uses a `liminf`.-/
theorem coverEntropyInf_image_of_comap (u : UniformSpace Y) {S : X → X} {T : Y → Y} {φ : X → Y}
    (h : Semiconj φ S T) (F : Set X) :
    coverEntropyInf T (φ '' F) = @coverEntropyInf X (comap φ u) S F := by
  /-
    X : Type u_1
    Y : Type u_2
    u : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F : Set X
    ⊢ Eq (Dynamics.coverEntropyInf T (Set.image φ F)) (Dynamics.coverEntropyInf S F)
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      ⊢ LE.le (Dynamics.coverEntropyInf T (Set.image φ F)) (Dynamics.coverEntropyInf …
    -/
  · refine iSup₂_le fun V V_uni ↦ (coverEntropyInfEntourage_image_le h F V).trans ?_
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage S F (Set.preimage (Prod.map φ φ) V) …
    -/
    apply @coverEntropyInfEntourage_le_coverEntropyInf X (comap φ u) S F
    /-
      case a.h
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ Membership.mem (uniformity X) (Set.preimage (Prod.map φ φ) V)
    -/
    rw [uniformity_comap φ, mem_comap]
    /-
      case a.h
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      ⊢ Exists fun t => And (Membership.mem (uniformity Y) t) (HasSubset.Subset (Set …
    -/
    exact ⟨V, V_uni, Subset.rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      ⊢ LE.le (Dynamics.coverEntropyInf S F) (Dynamics.coverEntropyInf T (Set.image  …
    -/
  · refine iSup₂_le fun U U_uni ↦ ?_
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage S F U) (Dynamics.coverEntropyInf T  …
    -/
    simp only [uniformity_comap φ, mem_comap] at U_uni
    /-
      case a
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      U_uni : Exists fun t => And (Membership.mem (uniformity Y) t) (HasSubset.Subse …
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage S F U) (Dynamics.coverEntropyInf T  …
    -/
    rcases U_uni with ⟨V, V_uni, V_sub⟩
    /-
      case a.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage S F U) (Dynamics.coverEntropyInf T  …
    -/
    rcases comp_symm_mem_uniformity_sets V_uni with ⟨W, W_uni, W_symm, W_V⟩
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage S F U) (Dynamics.coverEntropyInf T  …
    -/
    apply (coverEntropyInfEntourage_antitone S F ((preimage_mono W_V).trans V_sub)).trans
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le ((fun U => Dynamics.coverEntropyInfEntourage S F U) (Set.preimage (Pro …
    -/
    apply (le_coverEntropyInfEntourage_image h F W_symm).trans
    /-
      case a.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      u : UniformSpace Y
      S : X → X
      T : Y → Y
      φ : X → Y
      h : Function.Semiconj φ S T
      F : Set X
      U : Set (Prod X X)
      V : Set (Prod Y Y)
      V_uni : Membership.mem (uniformity Y) V
      V_sub : HasSubset.Subset (Set.preimage (Prod.map φ φ) V) U
      W : Set (Prod Y Y)
      W_uni : Membership.mem (uniformity Y) W
      W_symm : SymmetricRel W
      W_V : HasSubset.Subset (compRel W W) V
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage T (Set.image φ F) W) (Dynamics.cove …
    -/
    exact coverEntropyInfEntourage_le_coverEntropyInf T (φ '' F) W_uni
    /-
      🎉 no goals
    -/


lemma coverEntropy_restrict_subset [UniformSpace X] {T : X → X} {F G : Set X} (hF : F ⊆ G)
    (hG : MapsTo T G G) :
    coverEntropy (hG.restrict T G G) (val ⁻¹' F) = coverEntropy T F := by
  rw [← coverEntropy_image_of_comap _ hG.val_restrict_apply (val ⁻¹' F), image_preimage_coe G F,
    inter_eq_right.2 hF]


lemma coverEntropyInf_restrict_subset [UniformSpace X] {T : X → X} {F G : Set X} (hF : F ⊆ G)
    (hG : MapsTo T G G) :
    coverEntropyInf (hG.restrict T G G) (val ⁻¹' F) = coverEntropyInf T F := by
  rw [← coverEntropyInf_image_of_comap _ hG.val_restrict_apply (val ⁻¹' F), image_preimage_coe G F,
    inter_eq_right.2 hF]


/-- The entropy of the restriction of `T` to an invariant set `F` is `coverEntropy S F`. This
theorem justifies our definition of `coverEntropy T F`.-/
theorem coverEntropy_restrict [UniformSpace X] {T : X → X} {F : Set X} (h : MapsTo T F F) :
    coverEntropy (h.restrict T F F) univ = coverEntropy T F := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    F : Set X
    h : Set.MapsTo T F F
    ⊢ Eq (Dynamics.coverEntropy (Set.MapsTo.restrict T F F h) Set.univ) (Dynamics. …
  -/
  rw [← coverEntropy_restrict_subset Subset.rfl h, coe_preimage_self F]
  /-
    🎉 no goals
  -/


/-- The entropy of `φ '' F` is lower than entropy of `F` if  `φ` is uniformly continuous.-/
theorem coverEntropy_image_le_of_uniformContinuous [UniformSpace X] [UniformSpace Y] {S : X → X}
    {T : Y → Y} {φ : X → Y} (h : Semiconj φ S T) (h' : UniformContinuous φ) (F : Set X) :
    coverEntropy T (φ '' F) ≤ coverEntropy S F := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    h' : UniformContinuous φ
    F : Set X
    ⊢ LE.le (Dynamics.coverEntropy T (Set.image φ F)) (Dynamics.coverEntropy S F)
  -/
  rw [coverEntropy_image_of_comap _ h F]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    h' : UniformContinuous φ
    F : Set X
    ⊢ LE.le (Dynamics.coverEntropy S F) (Dynamics.coverEntropy S F)
  -/
  exact coverEntropy_antitone S F (uniformContinuous_iff.1 h')
  /-
    🎉 no goals
  -/


/-- The entropy of `φ '' F` is lower than entropy of `F` if  `φ` is uniformly continuous. This
  version uses a `liminf`.-/
theorem coverEntropyInf_image_le_of_uniformContinuous [UniformSpace X] [UniformSpace Y] {S : X → X}
    {T : Y → Y} {φ : X → Y} (h : Semiconj φ S T) (h' : UniformContinuous φ) (F : Set X) :
    coverEntropyInf T (φ '' F) ≤ coverEntropyInf S F := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    h' : UniformContinuous φ
    F : Set X
    ⊢ LE.le (Dynamics.coverEntropyInf T (Set.image φ F)) (Dynamics.coverEntropyInf …
  -/
  rw [coverEntropyInf_image_of_comap _ h F]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    h' : UniformContinuous φ
    F : Set X
    ⊢ LE.le (Dynamics.coverEntropyInf S F) (Dynamics.coverEntropyInf S F)
  -/
  exact coverEntropyInf_antitone S F (uniformContinuous_iff.1 h')
  /-
    🎉 no goals
  -/


lemma coverEntropy_image_le_of_uniformContinuousOn_invariant [UniformSpace X] [UniformSpace Y]
    {S : X → X} {T : Y → Y} {φ : X → Y} (h : Semiconj φ S T) {F G : Set X}
    (h' : UniformContinuousOn φ G) (hF : F ⊆ G) (hG : MapsTo S G G) :
    coverEntropy T (φ '' F) ≤ coverEntropy S F := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F G : Set X
    h' : UniformContinuousOn φ G
    hF : HasSubset.Subset F G
    hG : Set.MapsTo S G G
    ⊢ LE.le (Dynamics.coverEntropy T (Set.image φ F)) (Dynamics.coverEntropy S F)
  -/
  rw [← coverEntropy_restrict_subset hF hG]
  have hφ : Semiconj (G.restrict φ) (hG.restrict S G G) T := by
    intro x
    rw [G.restrict_apply, G.restrict_apply, hG.val_restrict_apply, h.eq x]
  apply (coverEntropy_image_le_of_uniformContinuous hφ
    (uniformContinuousOn_iff_restrict.1 h') (val ⁻¹' F)).trans_eq'
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F G : Set X
    h' : UniformContinuousOn φ G
    hF : HasSubset.Subset F G
    hG : Set.MapsTo S G G
    hφ : Function.Semiconj (G.restrict φ) (Set.MapsTo.restrict S G G hG) T
    ⊢ Eq (Dynamics.coverEntropy T (Set.image φ F)) (Dynamics.coverEntropy T (Set.i …
  -/
  rw [← image_image_val_eq_restrict_image, image_preimage_coe G F, inter_eq_right.2 hF]
  /-
    🎉 no goals
  -/


lemma coverEntropyInf_image_le_of_uniformContinuousOn_invariant [UniformSpace X] [UniformSpace Y]
    {S : X → X} {T : Y → Y} {φ : X → Y} (h : Semiconj φ S T) {F G : Set X}
    (h' : UniformContinuousOn φ G) (hF : F ⊆ G) (hG : MapsTo S G G) :
    coverEntropyInf T (φ '' F) ≤ coverEntropyInf S F := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F G : Set X
    h' : UniformContinuousOn φ G
    hF : HasSubset.Subset F G
    hG : Set.MapsTo S G G
    ⊢ LE.le (Dynamics.coverEntropyInf T (Set.image φ F)) (Dynamics.coverEntropyInf …
  -/
  rw [← coverEntropyInf_restrict_subset hF hG]
  have hφ : Semiconj (G.restrict φ) (hG.restrict S G G) T := by
    intro a
    rw [G.restrict_apply, G.restrict_apply, hG.val_restrict_apply, h.eq a]
  apply (coverEntropyInf_image_le_of_uniformContinuous hφ
    (uniformContinuousOn_iff_restrict.1 h') (val ⁻¹' F)).trans_eq'
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : UniformSpace X
    inst✝ : UniformSpace Y
    S : X → X
    T : Y → Y
    φ : X → Y
    h : Function.Semiconj φ S T
    F G : Set X
    h' : UniformContinuousOn φ G
    hF : HasSubset.Subset F G
    hG : Set.MapsTo S G G
    hφ : Function.Semiconj (G.restrict φ) (Set.MapsTo.restrict S G G hG) T
    ⊢ Eq (Dynamics.coverEntropyInf T (Set.image φ F)) (Dynamics.coverEntropyInf T  …
  -/
  rw [← image_image_val_eq_restrict_image, image_preimage_coe G F, inter_eq_right.2 hF]
  /-
    🎉 no goals
  -/


