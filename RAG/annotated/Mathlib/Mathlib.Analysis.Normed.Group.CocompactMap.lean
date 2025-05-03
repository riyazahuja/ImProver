theorem CocompactMapClass.norm_le [ProperSpace F] [FunLike 𝓕 E F] [CocompactMapClass 𝓕 E F]
    (ε : ℝ) : ∃ r : ℝ, ∀ x : E, r < ‖x‖ → ε < ‖f x‖ := by
  /-
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  have h := cocompact_tendsto f
  /-
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Filter.Tendsto (⇑f) (Filter.cocompact E) (Filter.cocompact F)
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  rw [tendsto_def] at h
  /-
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : ∀ (s : Set F), Membership.mem (Filter.cocompact F) s → Membership.mem (Fil …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  specialize h (Metric.closedBall 0 ε)ᶜ (mem_cocompact_of_closedBall_compl_subset 0 ⟨ε, rfl.subset⟩)
  /-
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  rcases closedBall_compl_subset_of_mem_cocompact h 0 with ⟨r, hr⟩
  /-
    case intro
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  use r
  /-
    case h
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    ⊢ ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
  -/
  intro x hx
  /-
    case h
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hx : LT.lt r (Norm.norm x)
    ⊢ LT.lt ε (Norm.norm (f x))
  -/
  suffices x ∈ f⁻¹' (Metric.closedBall 0 ε)ᶜ by aesop
  /-
    case h
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hx : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (Set.preimage (⇑f) (HasCompl.compl (Metric.closedBall 0 ε))) x
  -/
  apply hr
  /-
    case h.a
    E : Type u_2
    F : Type u_3
    𝓕 : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    f : 𝓕
    inst✝² : ProperSpace F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : CocompactMapClass 𝓕 E F
    ε : Real
    h : Membership.mem (Filter.cocompact E) (Set.preimage (⇑f) (HasCompl.compl (Me …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hx : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (HasCompl.compl (Metric.closedBall 0 r)) x
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


theorem Filter.tendsto_cocompact_cocompact_of_norm [ProperSpace E] {f : E → F}
    (h : ∀ ε : ℝ, ∃ r : ℝ, ∀ x : E, r < ‖x‖ → ε < ‖f x‖) :
    Tendsto f (cocompact E) (cocompact F) := by
  /-
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    ⊢ Filter.Tendsto f (Filter.cocompact E) (Filter.cocompact F)
  -/
  rw [tendsto_def]
  /-
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    ⊢ ∀ (s : Set F), Membership.mem (Filter.cocompact F) s → Membership.mem (Filte …
  -/
  intro s hs
  /-
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ⊢ Membership.mem (Filter.cocompact E) (Set.preimage f s)
  -/
  rcases closedBall_compl_subset_of_mem_cocompact hs 0 with ⟨ε, hε⟩
  /-
    case intro
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    ⊢ Membership.mem (Filter.cocompact E) (Set.preimage f s)
  -/
  rcases h ε with ⟨r, hr⟩
  /-
    case intro.intro
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    ⊢ Membership.mem (Filter.cocompact E) (Set.preimage f s)
  -/
  apply mem_cocompact_of_closedBall_compl_subset 0
  /-
    case intro.intro
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    ⊢ Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (S …
  -/
  use r
  /-
    case h
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    ⊢ HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage f s)
  -/
  intro x hx
  /-
    case h
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    x : E
    hx : Membership.mem (HasCompl.compl (Metric.closedBall 0 r)) x
    ⊢ Membership.mem (Set.preimage f s) x
  -/
  simp only [Set.mem_compl_iff, Metric.mem_closedBall, dist_zero_right, not_le] at hx
  /-
    case h
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    x : E
    hx : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (Set.preimage f s) x
  -/
  apply hε
  /-
    case h.a
    E : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε ( …
    s : Set F
    hs : Membership.mem (Filter.cocompact F) s
    ε : Real
    hε : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 ε)) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt ε (Norm.norm (f x))
    x : E
    hx : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (HasCompl.compl (Metric.closedBall 0 ε)) (f x)
  -/
  simp [hr x hx]
  /-
    🎉 no goals
  -/


theorem ContinuousMapClass.toCocompactMapClass_of_norm [ProperSpace E] [FunLike 𝓕 E F]
    [ContinuousMapClass 𝓕 E F] (h : ∀ (f : 𝓕) (ε : ℝ), ∃ r : ℝ, ∀ x : E, r < ‖x‖ → ε < ‖f x‖) :
    CocompactMapClass 𝓕 E F where
  cocompact_tendsto := (tendsto_cocompact_cocompact_of_norm <| h ·)

