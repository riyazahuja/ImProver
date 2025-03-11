/-- The second intersection of a sphere with a line through a point on that sphere; that point
if it is the only point of intersection of the line with the sphere. The intended use of this
definition is when `p ∈ s`; the definition does not use `s.radius`, so in general it returns
the second intersection with the sphere through `p` and with center `s.center`. -/
def Sphere.secondInter (s : Sphere P) (p : P) (v : V) : P :=
  (-2 * ⟪v, p -ᵥ s.center⟫ / ⟪v, v⟫) • v +ᵥ p


/-- The distance between `secondInter` and the center equals the distance between the original
point and the center. -/
@[simp]
theorem Sphere.secondInter_dist (s : Sphere P) (p : P) (v : V) :
    dist (s.secondInter p v) s.center = dist p s.center := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Eq (Dist.dist (s.secondInter p v) s.center) (Dist.dist p s.center)
  -/
  rw [Sphere.secondInter]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul (HDiv.hDiv (HMul.hMul (-2) (Inner.in …
  -/
  by_cases hv : v = 0; · simp [hv]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    ⊢ Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul (HDiv.hDiv (HMul.hMul (-2) (Inner.in …
  -/
  rw [dist_smul_vadd_eq_dist _ _ hv]
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    ⊢ Or (Eq (HDiv.hDiv (HMul.hMul (-2) (Inner.inner v (VSub.vsub p s.center))) (I …
  -/
  exact Or.inr rfl
  /-
    🎉 no goals
  -/


/-- The point given by `secondInter` lies on the sphere. -/
@[simp]
theorem Sphere.secondInter_mem {s : Sphere P} {p : P} (v : V) : s.secondInter p v ∈ s ↔ p ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Iff (Membership.mem s (s.secondInter p v)) (Membership.mem s p)
  -/
  simp_rw [mem_sphere, Sphere.secondInter_dist]
  /-
    🎉 no goals
  -/


/-- If the vector is zero, `secondInter` gives the original point. -/
@[simp]
theorem Sphere.secondInter_zero (s : Sphere P) (p : P) : s.secondInter p (0 : V) = p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    ⊢ Eq (s.secondInter p 0) p
  -/
  simp [Sphere.secondInter]
  /-
    🎉 no goals
  -/


/-- The point given by `secondInter` equals the original point if and only if the line is
orthogonal to the radius vector. -/
theorem Sphere.secondInter_eq_self_iff {s : Sphere P} {p : P} {v : V} :
    s.secondInter p v = p ↔ ⟪v, p -ᵥ s.center⟫ = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Iff (Eq (s.secondInter p v) p) (Eq (Inner.inner v (VSub.vsub p s.center)) 0)
  -/
  refine ⟨fun hp => ?_, fun hp => ?_⟩
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      v : V
      hp : Eq (s.secondInter p v) p
      ⊢ Eq (Inner.inner v (VSub.vsub p s.center)) 0
    -/
  · by_cases hv : v = 0
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p : P
        v : V
        hp : Eq (s.secondInter p v) p
        hv : Eq v 0
        ⊢ Eq (Inner.inner v (VSub.vsub p s.center)) 0
      -/
    · simp [hv]
      /-
        🎉 no goals
      -/
    rwa [Sphere.secondInter, eq_comm, eq_vadd_iff_vsub_eq, vsub_self, eq_comm, smul_eq_zero,
      or_iff_left hv, div_eq_zero_iff, inner_self_eq_zero, or_iff_left hv, mul_eq_zero,
      or_iff_right (by norm_num : (-2 : ℝ) ≠ 0)] at hp
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      v : V
      hp : Eq (Inner.inner v (VSub.vsub p s.center)) 0
      ⊢ Eq (s.secondInter p v) p
    -/
  · rw [Sphere.secondInter, hp, mul_zero, zero_div, zero_smul, zero_vadd]
    /-
      🎉 no goals
    -/


/-- A point on a line through a point on a sphere equals that point or `secondInter`. -/
theorem Sphere.eq_or_eq_secondInter_of_mem_mk'_span_singleton_iff_mem {s : Sphere P} {p : P}
    (hp : p ∈ s) {v : V} {p' : P} (hp' : p' ∈ AffineSubspace.mk' p (ℝ ∙ v)) :
    p' = p ∨ p' = s.secondInter p v ↔ p' ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    hp : Membership.mem s p
    v : V
    p' : P
    hp' : Membership.mem (AffineSubspace.mk' p (Submodule.span Real (Singleton.sin …
    ⊢ Iff (Or (Eq p' p) (Eq p' (s.secondInter p v))) (Membership.mem s p')
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      p' : P
      hp' : Membership.mem (AffineSubspace.mk' p (Submodule.span Real (Singleton.sin …
      h : Or (Eq p' p) (Eq p' (s.secondInter p v))
      ⊢ Membership.mem s p'
    -/
  · rcases h with (h | h)
      /-
        case refine_1.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p : P
        hp : Membership.mem s p
        v : V
        p' : P
        hp' : Membership.mem (AffineSubspace.mk' p (Submodule.span Real (Singleton.sin …
        h : Eq p' p
        ⊢ Membership.mem s p'
      -/
    · rwa [h]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p : P
        hp : Membership.mem s p
        v : V
        p' : P
        hp' : Membership.mem (AffineSubspace.mk' p (Submodule.span Real (Singleton.sin …
        h : Eq p' (s.secondInter p v)
        ⊢ Membership.mem s p'
      -/
    · rwa [h, Sphere.secondInter_mem]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      p' : P
      hp' : Membership.mem (AffineSubspace.mk' p (Submodule.span Real (Singleton.sin …
      h : Membership.mem s p'
      ⊢ Or (Eq p' p) (Eq p' (s.secondInter p v))
    -/
  · rw [AffineSubspace.mem_mk'_iff_vsub_mem, Submodule.mem_span_singleton] at hp'
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      p' : P
      hp' : Exists fun a => Eq (HSMul.hSMul a v) (VSub.vsub p' p)
      h : Membership.mem s p'
      ⊢ Or (Eq p' p) (Eq p' (s.secondInter p v))
    -/
    rcases hp' with ⟨r, hr⟩
    /-
      case refine_2.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      p' : P
      h : Membership.mem s p'
      r : Real
      hr : Eq (HSMul.hSMul r v) (VSub.vsub p' p)
      ⊢ Or (Eq p' p) (Eq p' (s.secondInter p v))
    -/
    rw [eq_comm, ← eq_vadd_iff_vsub_eq] at hr
    /-
      case refine_2.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      p' : P
      h : Membership.mem s p'
      r : Real
      hr : Eq p' (HVAdd.hVAdd (HSMul.hSMul r v) p)
      ⊢ Or (Eq p' p) (Eq p' (s.secondInter p v))
    -/
    subst hr
    /-
      case refine_2.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      r : Real
      h : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
      ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
    -/
    by_cases hv : v = 0
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p : P
        hp : Membership.mem s p
        v : V
        r : Real
        h : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
        hv : Eq v 0
        ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
      -/
    · simp [hv]
      /-
        🎉 no goals
      -/
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      r : Real
      h : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
      hv : Not (Eq v 0)
      ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
    -/
    rw [Sphere.secondInter]
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Membership.mem s p
      v : V
      r : Real
      h : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul r v) p)
      hv : Not (Eq v 0)
      ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
    -/
    rw [mem_sphere] at h hp
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Eq (Dist.dist p s.center) s.radius
      v : V
      r : Real
      h : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul r v) p) s.center) s.radius
      hv : Not (Eq v 0)
      ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
    -/
    rw [← hp, dist_smul_vadd_eq_dist _ _ hv] at h
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p : P
      hp : Eq (Dist.dist p s.center) s.radius
      v : V
      r : Real
      h : Or (Eq r 0) (Eq r (HDiv.hDiv (HMul.hMul (-2) (Inner.inner v (VSub.vsub p s …
      hv : Not (Eq v 0)
      ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul r v) p) p) (Eq (HVAdd.hVAdd (HSMul.hSMul r  …
    -/
                              /-
                                🎉 no goals
                              -/
    rcases h with (h | h) <;> simp [h]
                              /-
                                🎉 no goals
                              -/


/-- `secondInter` is unchanged by multiplying the vector by a nonzero real. -/
@[simp]
theorem Sphere.secondInter_smul (s : Sphere P) (p : P) (v : V) {r : ℝ} (hr : r ≠ 0) :
    s.secondInter p (r • v) = s.secondInter p v := by
  simp_rw [Sphere.secondInter, real_inner_smul_left, inner_smul_right, smul_smul,
    div_mul_eq_div_div]
  rw [mul_comm, ← mul_div_assoc, ← mul_div_assoc, mul_div_cancel_left₀ _ hr, mul_comm, mul_assoc,
    mul_div_cancel_left₀ _ hr, mul_comm]


/-- `secondInter` is unchanged by negating the vector. -/
@[simp]
theorem Sphere.secondInter_neg (s : Sphere P) (p : P) (v : V) :
    s.secondInter p (-v) = s.secondInter p v := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Eq (s.secondInter p (Neg.neg v)) (s.secondInter p v)
  -/
  rw [← neg_one_smul ℝ v, s.secondInter_smul p v (by norm_num : (-1 : ℝ) ≠ 0)]
  /-
    🎉 no goals
  -/


/-- Applying `secondInter` twice returns the original point. -/
@[simp]
theorem Sphere.secondInter_secondInter (s : Sphere P) (p : P) (v : V) :
    s.secondInter (s.secondInter p v) v = p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    ⊢ Eq (s.secondInter (s.secondInter p v) v) p
  -/
  by_cases hv : v = 0; · simp [hv]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    ⊢ Eq (s.secondInter (s.secondInter p v) v) p
  -/
  have hv' : ⟪v, v⟫ ≠ 0 := inner_self_ne_zero.2 hv
  simp only [Sphere.secondInter, vadd_vsub_assoc, vadd_vadd, inner_add_right, inner_smul_right,
    div_mul_cancel₀ _ hv']
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    hv' : Ne (Inner.inner v v) 0
    ⊢ Eq (HVAdd.hVAdd (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HMul.hMul (-2) (HAdd.hAd …
  -/
  rw [← @vsub_eq_zero_iff_eq V, vadd_vsub, ← add_smul, ← add_div]
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    hv' : Ne (Inner.inner v v) 0
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul (-2) (HAdd.hAdd (HMul.hMul  …
  -/
  convert zero_smul ℝ (M := V) _
  /-
    case h.e'_2.h.e'_5
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    hv' : Ne (Inner.inner v v) 0
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul (-2) (HAdd.hAdd (HMul.hMul (-2) (Inner.i …
  -/
  convert zero_div (G₀ := ℝ) _
  /-
    case h.e'_2.h.e'_5
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p : P
    v : V
    hv : Not (Eq v 0)
    hv' : Ne (Inner.inner v v) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (-2) (HAdd.hAdd (HMul.hMul (-2) (Inner.inner v (VSu …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If the vector passed to `secondInter` is given by a subtraction involving the point in
`secondInter`, the result of `secondInter` may be expressed using `lineMap`. -/
theorem Sphere.secondInter_eq_lineMap (s : Sphere P) (p p' : P) :
    s.secondInter p (p' -ᵥ p) =
      AffineMap.lineMap p p' (-2 * ⟪p' -ᵥ p, p -ᵥ s.center⟫ / ⟪p' -ᵥ p, p' -ᵥ p⟫) :=
  rfl


/-- If the vector passed to `secondInter` is given by a subtraction involving the point in
`secondInter`, the result lies in the span of the two points. -/
theorem Sphere.secondInter_vsub_mem_affineSpan (s : Sphere P) (p₁ p₂ : P) :
    s.secondInter p₁ (p₂ -ᵥ p₁) ∈ line[ℝ, p₁, p₂] :=
  smul_vsub_vadd_mem_affineSpan_pair _ _ _


/-- If the vector passed to `secondInter` is given by a subtraction involving the point in
`secondInter`, the three points are collinear. -/
theorem Sphere.secondInter_collinear (s : Sphere P) (p p' : P) :
    Collinear ℝ ({p, p', s.secondInter p (p' -ᵥ p)} : Set P) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p p' : P
    ⊢ Collinear Real (Insert.insert p (Insert.insert p' (Singleton.singleton (s.se …
  -/
  rw [Set.pair_comm, Set.insert_comm]
  exact
    (collinear_insert_iff_of_mem_affineSpan (s.secondInter_vsub_mem_affineSpan _ _)).2
      (collinear_pair ℝ _ _)


/-- If the vector passed to `secondInter` is given by a subtraction involving the point in
`secondInter`, and the second point is not outside the sphere, the second point is weakly
between the first point and the result of `secondInter`. -/
theorem Sphere.wbtw_secondInter {s : Sphere P} {p p' : P} (hp : p ∈ s)
    (hp' : dist p' s.center ≤ s.radius) : Wbtw ℝ p p' (s.secondInter p (p' -ᵥ p)) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p p' : P
    hp : Membership.mem s p
    hp' : LE.le (Dist.dist p' s.center) s.radius
    ⊢ Wbtw Real p p' (s.secondInter p (VSub.vsub p' p))
  -/
  by_cases h : p' = p; · simp [h]
                         /-
                           🎉 no goals
                         -/
  refine
    wbtw_of_collinear_of_dist_center_le_radius (s.secondInter_collinear p p') hp hp'
      ((Sphere.secondInter_mem _).2 hp) ?_
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p p' : P
    hp : Membership.mem s p
    hp' : LE.le (Dist.dist p' s.center) s.radius
    h : Not (Eq p' p)
    ⊢ Ne p (s.secondInter p (VSub.vsub p' p))
  -/
  intro he
  rw [eq_comm, Sphere.secondInter_eq_self_iff, ← neg_neg (p' -ᵥ p), inner_neg_left,
    neg_vsub_eq_vsub_rev, neg_eq_zero, eq_comm] at he
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p p' : P
    hp : Membership.mem s p
    hp' : LE.le (Dist.dist p' s.center) s.radius
    h : Not (Eq p' p)
    he : Eq 0 (Inner.inner (VSub.vsub p p') (VSub.vsub p s.center))
    ⊢ False
  -/
  exact ((inner_pos_or_eq_of_dist_le_radius hp hp').resolve_right (Ne.symm h)).ne he
  /-
    🎉 no goals
  -/


/-- If the vector passed to `secondInter` is given by a subtraction involving the point in
`secondInter`, and the second point is inside the sphere, the second point is strictly between
the first point and the result of `secondInter`. -/
theorem Sphere.sbtw_secondInter {s : Sphere P} {p p' : P} (hp : p ∈ s)
    (hp' : dist p' s.center < s.radius) : Sbtw ℝ p p' (s.secondInter p (p' -ᵥ p)) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p p' : P
    hp : Membership.mem s p
    hp' : LT.lt (Dist.dist p' s.center) s.radius
    ⊢ Sbtw Real p p' (s.secondInter p (VSub.vsub p' p))
  -/
  refine ⟨Sphere.wbtw_secondInter hp hp'.le, ?_, ?_⟩
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p p' : P
      hp : Membership.mem s p
      hp' : LT.lt (Dist.dist p' s.center) s.radius
      ⊢ Ne p' p
    -/
  · rintro rfl
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p' : P
      hp' : LT.lt (Dist.dist p' s.center) s.radius
      hp : Membership.mem s p'
      ⊢ False
    -/
    rw [mem_sphere] at hp
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p' : P
      hp' : LT.lt (Dist.dist p' s.center) s.radius
      hp : Eq (Dist.dist p' s.center) s.radius
      ⊢ False
    -/
    simp [hp] at hp'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p p' : P
      hp : Membership.mem s p
      hp' : LT.lt (Dist.dist p' s.center) s.radius
      ⊢ Ne p' (s.secondInter p (VSub.vsub p' p))
    -/
  · rintro h
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p p' : P
      hp : Membership.mem s p
      hp' : LT.lt (Dist.dist p' s.center) s.radius
      h : Eq p' (s.secondInter p (VSub.vsub p' p))
      ⊢ False
    -/
    rw [h, mem_sphere.1 ((Sphere.secondInter_mem _).2 hp)] at hp'
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p p' : P
      hp : Membership.mem s p
      hp' : LT.lt s.radius s.radius
      h : Eq p' (s.secondInter p (VSub.vsub p' p))
      ⊢ False
    -/
    exact lt_irrefl _ hp'
    /-
      🎉 no goals
    -/


