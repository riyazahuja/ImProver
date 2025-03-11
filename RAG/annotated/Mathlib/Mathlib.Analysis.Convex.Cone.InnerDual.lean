/-- The dual cone is the cone consisting of all points `y` such that for
all points `x` in a given set `0 ≤ ⟪ x, y ⟫`. -/
def Set.innerDualCone (s : Set H) : ConvexCone ℝ H where
  carrier := { y | ∀ x ∈ s, 0 ≤ ⟪x, y⟫ }
  smul_mem' c hc y hy x hx := by
    /-
      H : Type u_1
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace Real H
      s✝ t s : Set H
      c : Real
      hc : LT.lt 0 c
      y : H
      hy : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      x : H
      hx : Membership.mem s x
      ⊢ LE.le 0 (Inner.inner x (HSMul.hSMul c y))
    -/
    rw [real_inner_smul_right]
    /-
      H : Type u_1
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace Real H
      s✝ t s : Set H
      c : Real
      hc : LT.lt 0 c
      y : H
      hy : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      x : H
      hx : Membership.mem s x
      ⊢ LE.le 0 (HMul.hMul c (Inner.inner x y))
    -/
    exact mul_nonneg hc.le (hy x hx)
    /-
      🎉 no goals
    -/
  add_mem' u hu v hv x hx := by
    /-
      H : Type u_1
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace Real H
      s✝ t s : Set H
      u : H
      hu : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      v : H
      hv : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      x : H
      hx : Membership.mem s x
      ⊢ LE.le 0 (Inner.inner x (HAdd.hAdd u v))
    -/
    rw [inner_add_right]
    /-
      H : Type u_1
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace Real H
      s✝ t s : Set H
      u : H
      hu : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      v : H
      hv : Membership.mem (setOf fun y => ∀ (x : H), Membership.mem s x → LE.le 0 (I …
      x : H
      hx : Membership.mem s x
      ⊢ LE.le 0 (HAdd.hAdd (Inner.inner x u) (Inner.inner x v))
    -/
    exact add_nonneg (hu x hx) (hv x hx)
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_innerDualCone (y : H) (s : Set H) : y ∈ s.innerDualCone ↔ ∀ x ∈ s, 0 ≤ ⟪x, y⟫ :=
  Iff.rfl


@[simp]
theorem innerDualCone_empty : (∅ : Set H).innerDualCone = ⊤ :=
  eq_top_iff.mpr fun _ _ _ => False.elim


/-- Dual cone of the convex cone {0} is the total space. -/
@[simp]
theorem innerDualCone_zero : (0 : Set H).innerDualCone = ⊤ :=
  eq_top_iff.mpr fun _ _ y (hy : y = 0) => hy.symm ▸ (inner_zero_left _).ge


/-- Dual cone of the total space is the convex cone {0}. -/
@[simp]
theorem innerDualCone_univ : (univ : Set H).innerDualCone = 0 := by
  suffices ∀ x : H, x ∈ (univ : Set H).innerDualCone → x = 0 by
    apply SetLike.coe_injective
    exact eq_singleton_iff_unique_mem.mpr ⟨fun x _ => (inner_zero_right _).ge, this⟩
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ⊢ ∀ (x : H), Membership.mem Set.univ.innerDualCone x → Eq x 0
  -/
  exact fun x hx => by simpa [← real_inner_self_nonpos] using hx (-x) (mem_univ _)
  /-
    🎉 no goals
  -/


theorem innerDualCone_le_innerDualCone (h : t ⊆ s) : s.innerDualCone ≤ t.innerDualCone :=
  fun _ hy x hx => hy x (h hx)


                                                                         /-
                                                                           H : Type u_1
                                                                           inst✝¹ : NormedAddCommGroup H
                                                                           inst✝ : InnerProductSpace Real H
                                                                           s : Set H
                                                                           x : H
                                                                           x✝ : Membership.mem s x
                                                                           ⊢ LE.le 0 (Inner.inner x 0)
                                                                         -/
theorem pointed_innerDualCone : s.innerDualCone.Pointed := fun x _ => by rw [inner_zero_right]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The inner dual cone of a singleton is given by the preimage of the positive cone under the
linear map `fun y ↦ ⟪x, y⟫`. -/
theorem innerDualCone_singleton (x : H) :
    ({x} : Set H).innerDualCone = (ConvexCone.positive ℝ ℝ).comap (innerₛₗ ℝ x) :=
  ConvexCone.ext fun _ => forall_eq


theorem innerDualCone_union (s t : Set H) :
    (s ∪ t).innerDualCone = s.innerDualCone ⊓ t.innerDualCone :=
  le_antisymm (le_inf (fun _ hx _ hy => hx _ <| Or.inl hy) fun _ hx _ hy => hx _ <| Or.inr hy)
    fun _ hx _ => Or.rec (hx.1 _) (hx.2 _)


theorem innerDualCone_insert (x : H) (s : Set H) :
    (insert x s).innerDualCone = Set.innerDualCone {x} ⊓ s.innerDualCone := by
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    x : H
    s : Set H
    ⊢ Eq (Insert.insert x s).innerDualCone (Min.min (Singleton.singleton x).innerD …
  -/
  rw [insert_eq, innerDualCone_union]
  /-
    🎉 no goals
  -/


theorem innerDualCone_iUnion {ι : Sort*} (f : ι → Set H) :
    (⋃ i, f i).innerDualCone = ⨅ i, (f i).innerDualCone := by
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ι : Sort u_2
    f : ι → Set H
    ⊢ Eq (Set.iUnion fun i => f i).innerDualCone (iInf fun i => (f i).innerDualCone)
  -/
  refine le_antisymm (le_iInf fun i x hx y hy => hx _ <| mem_iUnion_of_mem _ hy) ?_
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ι : Sort u_2
    f : ι → Set H
    ⊢ LE.le (iInf fun i => (f i).innerDualCone) (Set.iUnion fun i => f i).innerDua …
  -/
  intro x hx y hy
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ι : Sort u_2
    f : ι → Set H
    x : H
    hx : Membership.mem (iInf fun i => (f i).innerDualCone) x
    y : H
    hy : Membership.mem (Set.iUnion fun i => f i) y
    ⊢ LE.le 0 (Inner.inner y x)
  -/
  rw [ConvexCone.mem_iInf] at hx
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ι : Sort u_2
    f : ι → Set H
    x : H
    hx : ∀ (i : ι), Membership.mem (f i).innerDualCone x
    y : H
    hy : Membership.mem (Set.iUnion fun i => f i) y
    ⊢ LE.le 0 (Inner.inner y x)
  -/
  obtain ⟨j, hj⟩ := mem_iUnion.mp hy
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    ι : Sort u_2
    f : ι → Set H
    x : H
    hx : ∀ (i : ι), Membership.mem (f i).innerDualCone x
    y : H
    hy : Membership.mem (Set.iUnion fun i => f i) y
    j : ι
    hj : Membership.mem (f j) y
    ⊢ LE.le 0 (Inner.inner y x)
  -/
  exact hx _ _ hj
  /-
    🎉 no goals
  -/


theorem innerDualCone_sUnion (S : Set (Set H)) :
    (⋃₀ S).innerDualCone = sInf (Set.innerDualCone '' S) := by
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    S : Set (Set H)
    ⊢ Eq S.sUnion.innerDualCone (InfSet.sInf (Set.image Set.innerDualCone S))
  -/
  simp_rw [sInf_image, sUnion_eq_biUnion, innerDualCone_iUnion]
  /-
    🎉 no goals
  -/


/-- The dual cone of `s` equals the intersection of dual cones of the points in `s`. -/
theorem innerDualCone_eq_iInter_innerDualCone_singleton :
    (s.innerDualCone : Set H) = ⋂ i : s, (({↑i} : Set H).innerDualCone : Set H) := by
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    ⊢ Eq (↑s.innerDualCone) (Set.iInter fun i => ↑(Singleton.singleton ↑i).innerDu …
  -/
  rw [← ConvexCone.coe_iInf, ← innerDualCone_iUnion, iUnion_of_singleton_coe]
  /-
    🎉 no goals
  -/


theorem isClosed_innerDualCone : IsClosed (s.innerDualCone : Set H) := by
  -- reduce the problem to showing that dual cone of a singleton `{x}` is closed
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    ⊢ IsClosed ↑s.innerDualCone
  -/
  rw [innerDualCone_eq_iInter_innerDualCone_singleton]
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    ⊢ IsClosed (Set.iInter fun i => ↑(Singleton.singleton ↑i).innerDualCone)
  -/
  apply isClosed_iInter
  /-
    case h
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    ⊢ ∀ (i : ↑s), IsClosed ↑(Singleton.singleton ↑i).innerDualCone
  -/
  intro x
  -- the dual cone of a singleton `{x}` is the preimage of `[0, ∞)` under `inner x`
  have h : ({↑x} : Set H).innerDualCone = (inner x : H → ℝ) ⁻¹' Set.Ici 0 := by
    rw [innerDualCone_singleton, ConvexCone.coe_comap, ConvexCone.coe_positive, innerₛₗ_apply_coe]
  -- the preimage is closed as `inner x` is continuous and `[0, ∞)` is closed
  /-
    case h
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    x : ↑s
    h : Eq (↑(Singleton.singleton ↑x).innerDualCone) (Set.preimage (Inner.inner ↑x …
    ⊢ IsClosed ↑(Singleton.singleton ↑x).innerDualCone
  -/
  rw [h]
  /-
    case h
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    s : Set H
    x : ↑s
    h : Eq (↑(Singleton.singleton ↑x).innerDualCone) (Set.preimage (Inner.inner ↑x …
    ⊢ IsClosed (Set.preimage (Inner.inner ↑x) (Set.Ici 0))
  -/
  exact isClosed_Ici.preimage (continuous_const.inner continuous_id')
  /-
    🎉 no goals
  -/


theorem ConvexCone.pointed_of_nonempty_of_isClosed (K : ConvexCone ℝ H) (ne : (K : Set H).Nonempty)
    (hc : IsClosed (K : Set H)) : K.Pointed := by
  /-
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    ⊢ K.Pointed
  -/
  obtain ⟨x, hx⟩ := ne
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    hc : IsClosed ↑K
    x : H
    hx : Membership.mem (↑K) x
    ⊢ K.Pointed
  -/
  let f : ℝ → H := (· • x)
  -- f (0, ∞) is a subset of K
  have fI : f '' Set.Ioi 0 ⊆ (K : Set H) := by
    rintro _ ⟨_, h, rfl⟩
    exact K.smul_mem (Set.mem_Ioi.1 h) hx
  -- closure of f (0, ∞) is a subset of K
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    hc : IsClosed ↑K
    x : H
    hx : Membership.mem (↑K) x
    f : Real → H := fun x_1 => HSMul.hSMul x_1 x
    fI : HasSubset.Subset (Set.image f (Set.Ioi 0)) ↑K
    ⊢ K.Pointed
  -/
  have clf : closure (f '' Set.Ioi 0) ⊆ (K : Set H) := hc.closure_subset_iff.2 fI
  -- f is continuous at 0 from the right
  have fc : ContinuousWithinAt f (Set.Ioi (0 : ℝ)) 0 :=
    (continuous_id.smul continuous_const).continuousWithinAt
  -- 0 belongs to the closure of the f (0, ∞)
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    hc : IsClosed ↑K
    x : H
    hx : Membership.mem (↑K) x
    f : Real → H := fun x_1 => HSMul.hSMul x_1 x
    fI : HasSubset.Subset (Set.image f (Set.Ioi 0)) ↑K
    clf : HasSubset.Subset (closure (Set.image f (Set.Ioi 0))) ↑K
    fc : ContinuousWithinAt f (Set.Ioi 0) 0
    ⊢ K.Pointed
  -/
  have mem₀ := fc.mem_closure_image (by rw [closure_Ioi (0 : ℝ), mem_Ici])
  -- as 0 ∈ closure f (0, ∞) and closure f (0, ∞) ⊆ K, 0 ∈ K.
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    hc : IsClosed ↑K
    x : H
    hx : Membership.mem (↑K) x
    f : Real → H := fun x_1 => HSMul.hSMul x_1 x
    fI : HasSubset.Subset (Set.image f (Set.Ioi 0)) ↑K
    clf : HasSubset.Subset (closure (Set.image f (Set.Ioi 0))) ↑K
    fc : ContinuousWithinAt f (Set.Ioi 0) 0
    mem₀ : Membership.mem (closure (Set.image f (Set.Ioi 0))) (f 0)
    ⊢ K.Pointed
  -/
  have f₀ : f 0 = 0 := zero_smul ℝ x
  /-
    case intro
    H : Type u_1
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace Real H
    K : ConvexCone Real H
    hc : IsClosed ↑K
    x : H
    hx : Membership.mem (↑K) x
    f : Real → H := fun x_1 => HSMul.hSMul x_1 x
    fI : HasSubset.Subset (Set.image f (Set.Ioi 0)) ↑K
    clf : HasSubset.Subset (closure (Set.image f (Set.Ioi 0))) ↑K
    fc : ContinuousWithinAt f (Set.Ioi 0) 0
    mem₀ : Membership.mem (closure (Set.image f (Set.Ioi 0))) (f 0)
    f₀ : Eq (f 0) 0
    ⊢ K.Pointed
  -/
  simpa only [f₀, ConvexCone.Pointed, ← SetLike.mem_coe] using mem_of_subset_of_mem clf mem₀
  /-
    🎉 no goals
  -/


open scoped InnerProductSpace in
/-- This is a stronger version of the Hahn-Banach separation theorem for closed convex cones. This
is also the geometric interpretation of Farkas' lemma. -/
theorem ConvexCone.hyperplane_separation_of_nonempty_of_isClosed_of_nmem (K : ConvexCone ℝ H)
    (ne : (K : Set H).Nonempty) (hc : IsClosed (K : Set H)) {b : H} (disj : b ∉ K) :
    ∃ y : H, (∀ x : H, x ∈ K → 0 ≤ ⟪x, y⟫_ℝ) ∧ ⟪y, b⟫_ℝ < 0 := by
  -- let `z` be the point in `K` closest to `b`
  /-
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    b : H
    disj : Not (Membership.mem K b)
    ⊢ Exists fun y => And (∀ (x : H), Membership.mem K x → LE.le 0 (Inner.inner x  …
  -/
  obtain ⟨z, hzK, infi⟩ := exists_norm_eq_iInf_of_complete_convex ne hc.isComplete K.convex b
  -- for any `w` in `K`, we have `⟪b - z, w - z⟫_ℝ ≤ 0`
  /-
    case intro.intro
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    b : H
    disj : Not (Membership.mem K b)
    z : H
    hzK : Membership.mem (↑K) z
    infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
    ⊢ Exists fun y => And (∀ (x : H), Membership.mem K x → LE.le 0 (Inner.inner x  …
  -/
  have hinner := (norm_eq_iInf_iff_real_inner_le_zero K.convex hzK).1 infi
  -- set `y := z - b`
  /-
    case intro.intro
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    b : H
    disj : Not (Membership.mem K b)
    z : H
    hzK : Membership.mem (↑K) z
    infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
    hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
    ⊢ Exists fun y => And (∀ (x : H), Membership.mem K x → LE.le 0 (Inner.inner x  …
  -/
  use z - b
  /-
    case h
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    b : H
    disj : Not (Membership.mem K b)
    z : H
    hzK : Membership.mem (↑K) z
    infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
    hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
    ⊢ And (∀ (x : H), Membership.mem K x → LE.le 0 (Inner.inner x (HSub.hSub z b)) …
  -/
  constructor
  · -- the rest of the proof is a straightforward calculation
    /-
      case h.left
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      b : H
      disj : Not (Membership.mem K b)
      z : H
      hzK : Membership.mem (↑K) z
      infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
      hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
      ⊢ ∀ (x : H), Membership.mem K x → LE.le 0 (Inner.inner x (HSub.hSub z b))
    -/
    rintro x hxK
    /-
      case h.left
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      b : H
      disj : Not (Membership.mem K b)
      z : H
      hzK : Membership.mem (↑K) z
      infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
      hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
      x : H
      hxK : Membership.mem K x
      ⊢ LE.le 0 (Inner.inner x (HSub.hSub z b))
    -/
    specialize hinner _ (K.add_mem hxK hzK)
    rwa [add_sub_cancel_right, real_inner_comm, ← neg_nonneg, neg_eq_neg_one_mul,
      ← real_inner_smul_right, neg_smul, one_smul, neg_sub] at hinner
  · -- as `K` is closed and non-empty, it is pointed
    /-
      case h.right
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      b : H
      disj : Not (Membership.mem K b)
      z : H
      hzK : Membership.mem (↑K) z
      infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
      hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
      ⊢ LT.lt (Inner.inner (HSub.hSub z b) b) 0
    -/
    have hinner₀ := hinner 0 (K.pointed_of_nonempty_of_isClosed ne hc)
    -- the rest of the proof is a straightforward calculation
    /-
      case h.right
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      b : H
      disj : Not (Membership.mem K b)
      z : H
      hzK : Membership.mem (↑K) z
      infi : Eq (Norm.norm (HSub.hSub b z)) (iInf fun w => Norm.norm (HSub.hSub b ↑w))
      hinner : ∀ (w : H), Membership.mem (↑K) w → LE.le (Inner.inner (HSub.hSub b z) …
      hinner₀ : LE.le (Inner.inner (HSub.hSub b z) (HSub.hSub 0 z)) 0
      ⊢ LT.lt (Inner.inner (HSub.hSub z b) b) 0
    -/
    rw [zero_sub, inner_neg_right, Right.neg_nonpos_iff] at hinner₀
    have hbz : b - z ≠ 0 := by
      rw [sub_ne_zero]
      contrapose! hzK
      rwa [← hzK]
    rw [← neg_zero, lt_neg, ← neg_one_mul, ← real_inner_smul_left, smul_sub, neg_smul, one_smul,
      neg_smul, neg_sub_neg, one_smul]
    calc
      0 < ⟪b - z, b - z⟫_ℝ := lt_of_not_le ((Iff.not real_inner_self_nonpos).2 hbz)
      _ = ⟪b - z, b - z⟫_ℝ + 0 := (add_zero _).symm
      _ ≤ ⟪b - z, b - z⟫_ℝ + ⟪b - z, z⟫_ℝ := add_le_add rfl.ge hinner₀
      _ = ⟪b - z, b - z + z⟫_ℝ := (inner_add_right _ _ _).symm
      _ = ⟪b - z, b⟫_ℝ := by rw [sub_add_cancel]


/-- The inner dual of inner dual of a non-empty, closed convex cone is itself. -/
theorem ConvexCone.innerDualCone_of_innerDualCone_eq_self (K : ConvexCone ℝ H)
    (ne : (K : Set H).Nonempty) (hc : IsClosed (K : Set H)) :
    ((K : Set H).innerDualCone : Set H).innerDualCone = K := by
  /-
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    ⊢ Eq (↑(↑K).innerDualCone).innerDualCone K
  -/
  ext x
  /-
    case h
    H : Type u_1
    inst✝² : NormedAddCommGroup H
    inst✝¹ : InnerProductSpace Real H
    inst✝ : CompleteSpace H
    K : ConvexCone Real H
    ne : (↑K).Nonempty
    hc : IsClosed ↑K
    x : H
    ⊢ Iff (Membership.mem (↑(↑K).innerDualCone).innerDualCone x) (Membership.mem K …
  -/
  constructor
    /-
      case h.mp
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      ⊢ Membership.mem (↑(↑K).innerDualCone).innerDualCone x → Membership.mem K x
    -/
  · rw [mem_innerDualCone, ← SetLike.mem_coe]
    /-
      case h.mp
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      ⊢ (∀ (x_1 : H), Membership.mem (↑(↑K).innerDualCone) x_1 → LE.le 0 (Inner.inne …
    -/
    contrapose!
    /-
      case h.mp
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      ⊢ Not (Membership.mem (↑K) x) → Exists fun x_1 => And (Membership.mem (↑(↑K).i …
    -/
    exact K.hyperplane_separation_of_nonempty_of_isClosed_of_nmem ne hc
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      ⊢ Membership.mem K x → Membership.mem (↑(↑K).innerDualCone).innerDualCone x
    -/
  · rintro hxK y h
    /-
      case h.mpr
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      hxK : Membership.mem K x
      y : H
      h : Membership.mem (↑(↑K).innerDualCone) y
      ⊢ LE.le 0 (Inner.inner y x)
    -/
    specialize h x hxK
    /-
      case h.mpr
      H : Type u_1
      inst✝² : NormedAddCommGroup H
      inst✝¹ : InnerProductSpace Real H
      inst✝ : CompleteSpace H
      K : ConvexCone Real H
      ne : (↑K).Nonempty
      hc : IsClosed ↑K
      x : H
      hxK : Membership.mem K x
      y : H
      h : LE.le 0 (Inner.inner x y)
      ⊢ LE.le 0 (Inner.inner y x)
    -/
    rwa [real_inner_comm]
    /-
      🎉 no goals
    -/


