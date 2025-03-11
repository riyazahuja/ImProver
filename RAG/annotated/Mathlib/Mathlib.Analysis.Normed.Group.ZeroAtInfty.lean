theorem ZeroAtInftyContinuousMapClass.norm_le (f : 𝓕) (ε : ℝ) (hε : 0 < ε) :
    ∃ (r : ℝ), ∀ (x : E) (_hx : r < ‖x‖), ‖f x‖ < ε := by
  /-
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  have h := zero_at_infty f
  /-
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Filter.Tendsto (⇑f) (Filter.cocompact E) (nhds 0)
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero, tendsto_def] at h
  /-
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (s : Set Real), Membership.mem (nhds 0) s → Membership.mem (Filter.cocom …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  specialize h (Metric.ball 0 ε) (Metric.ball_mem_nhds 0 hε)
  /-
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  rcases Metric.closedBall_compl_subset_of_mem_cocompact h 0 with ⟨r, hr⟩
  /-
    case intro
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    ⊢ Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  use r
  /-
    case h
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    ⊢ ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
  -/
  intro x hr'
  /-
    case h
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hr' : LT.lt r (Norm.norm x)
    ⊢ LT.lt (Norm.norm (f x)) ε
  -/
  suffices x ∈ (fun x ↦ ‖f x‖) ⁻¹' Metric.ball 0 ε by aesop
  /-
    case h
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hr' : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (Set.preimage (fun x => Norm.norm (f x)) (Metric.ball 0 ε)) x
  -/
  apply hr
  /-
    case h.a
    E : Type u_1
    F : Type u_2
    𝓕 : Type u_3
    inst✝³ : SeminormedAddGroup E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : FunLike 𝓕 E F
    inst✝ : ZeroAtInftyContinuousMapClass 𝓕 E F
    f : 𝓕
    ε : Real
    hε : LT.lt 0 ε
    h : Membership.mem (Filter.cocompact E) (Set.preimage (fun x => Norm.norm (f x …
    r : Real
    hr : HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage ( …
    x : E
    hr' : LT.lt r (Norm.norm x)
    ⊢ Membership.mem (HasCompl.compl (Metric.closedBall 0 r)) x
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem zero_at_infty_of_norm_le (f : E → F)
    (h : ∀ (ε : ℝ) (_hε : 0 < ε), ∃ (r : ℝ), ∀ (x : E) (_hx : r < ‖x‖), ‖f x‖ < ε) :
    Tendsto f (cocompact E) (𝓝 0) := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    ⊢ Filter.Tendsto f (Filter.cocompact E) (nhds 0)
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  /-
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    ⊢ Filter.Tendsto (fun x => Norm.norm (f x)) (Filter.cocompact E) (nhds 0)
  -/
  intro s hs
  /-
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    hs : Membership.mem (nhds 0) s
    ⊢ Membership.mem (Filter.map (fun x => Norm.norm (f x)) (Filter.cocompact E)) s
  -/
  rw [mem_map, Metric.mem_cocompact_iff_closedBall_compl_subset 0]
  /-
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    hs : Membership.mem (nhds 0) s
    ⊢ Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (S …
  -/
  rw [Metric.mem_nhds_iff] at hs
  /-
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    hs : Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball 0 ε) s)
    ⊢ Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (S …
  -/
  rcases hs with ⟨ε, hε, hs⟩
  /-
    case intro.intro
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    ε : Real
    hε : GT.gt ε 0
    hs : HasSubset.Subset (Metric.ball 0 ε) s
    ⊢ Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (S …
  -/
  rcases h ε hε with ⟨r, hr⟩
  /-
    case intro.intro.intro
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    ε : Real
    hε : GT.gt ε 0
    hs : HasSubset.Subset (Metric.ball 0 ε) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
    ⊢ Exists fun r => HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (S …
  -/
  use r
  /-
    case h
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    ε : Real
    hε : GT.gt ε 0
    hs : HasSubset.Subset (Metric.ball 0 ε) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
    ⊢ HasSubset.Subset (HasCompl.compl (Metric.closedBall 0 r)) (Set.preimage (fun …
  -/
  intro
  /-
    case h
    E : Type u_1
    F : Type u_2
    inst✝² : SeminormedAddGroup E
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : ProperSpace E
    f : E → F
    h : ∀ (ε : Real), LT.lt 0 ε → Exists fun r => ∀ (x : E), LT.lt r (Norm.norm x) …
    s : Set Real
    ε : Real
    hε : GT.gt ε 0
    hs : HasSubset.Subset (Metric.ball 0 ε) s
    r : Real
    hr : ∀ (x : E), LT.lt r (Norm.norm x) → LT.lt (Norm.norm (f x)) ε
    a✝ : E
    ⊢ Membership.mem (HasCompl.compl (Metric.closedBall 0 r)) a✝ → Membership.mem  …
  -/
  aesop
  /-
    🎉 no goals
  -/

