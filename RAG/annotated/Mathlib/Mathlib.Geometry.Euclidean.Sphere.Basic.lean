/-- A `Sphere P` bundles a `center` and `radius`. This definition does not require the radius to
be positive; that should be given as a hypothesis to lemmas that require it. -/
@[ext]
structure Sphere [MetricSpace P] where
  /-- center of this sphere -/
  center : P
  /-- radius of the sphere: not required to be positive -/
  radius : ℝ


instance [Nonempty P] : Nonempty (Sphere P) :=
  ⟨⟨Classical.arbitrary P, 0⟩⟩


instance : Coe (Sphere P) (Set P) :=
  ⟨fun s => Metric.sphere s.center s.radius⟩


instance : Membership P (Sphere P) :=
  ⟨fun s p => p ∈ (s : Set P)⟩


theorem Sphere.mk_center (c : P) (r : ℝ) : (⟨c, r⟩ : Sphere P).center = c :=
  rfl


theorem Sphere.mk_radius (c : P) (r : ℝ) : (⟨c, r⟩ : Sphere P).radius = r :=
  rfl


@[simp]
theorem Sphere.mk_center_radius (s : Sphere P) : (⟨s.center, s.radius⟩ : Sphere P) = s := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    s : EuclideanGeometry.Sphere P
    ⊢ Eq { center := s.center, radius := s.radius } s
  -/
          /-
            🎉 no goals
          -/
  ext <;> rfl
          /-
            🎉 no goals
          -/

/- Porting note: is a syntactic tautology
theorem Sphere.coe_def (s : Sphere P) : (s : Set P) = Metric.sphere s.center s.radius :=
  rfl -/


@[simp]
theorem Sphere.coe_mk (c : P) (r : ℝ) : ↑(⟨c, r⟩ : Sphere P) = Metric.sphere c r :=
  rfl

-- @[simp] -- Porting note: simp-normal form is `Sphere.mem_coe'`

theorem Sphere.mem_coe {p : P} {s : Sphere P} : p ∈ (s : Set P) ↔ p ∈ s :=
  Iff.rfl


@[simp]
theorem Sphere.mem_coe' {p : P} {s : Sphere P} : dist p s.center = s.radius ↔ p ∈ s :=
  Iff.rfl


theorem mem_sphere {p : P} {s : Sphere P} : p ∈ s ↔ dist p s.center = s.radius :=
  Iff.rfl


theorem mem_sphere' {p : P} {s : Sphere P} : p ∈ s ↔ dist s.center p = s.radius :=
  Metric.mem_sphere'


theorem subset_sphere {ps : Set P} {s : Sphere P} : ps ⊆ s ↔ ∀ p ∈ ps, p ∈ s :=
  Iff.rfl


theorem dist_of_mem_subset_sphere {p : P} {ps : Set P} {s : Sphere P} (hp : p ∈ ps)
    (hps : ps ⊆ (s : Set P)) : dist p s.center = s.radius :=
  mem_sphere.1 (Sphere.mem_coe.1 (Set.mem_of_mem_of_subset hp hps))


theorem dist_of_mem_subset_mk_sphere {p c : P} {ps : Set P} {r : ℝ} (hp : p ∈ ps)
    (hps : ps ⊆ ↑(⟨c, r⟩ : Sphere P)) : dist p c = r :=
  dist_of_mem_subset_sphere hp hps


theorem Sphere.ne_iff {s₁ s₂ : Sphere P} :
    s₁ ≠ s₂ ↔ s₁.center ≠ s₂.center ∨ s₁.radius ≠ s₂.radius := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    s₁ s₂ : EuclideanGeometry.Sphere P
    ⊢ Iff (Ne s₁ s₂) (Or (Ne s₁.center s₂.center) (Ne s₁.radius s₂.radius))
  -/
  rw [← not_and_or, ← Sphere.ext_iff]
  /-
    🎉 no goals
  -/


theorem Sphere.center_eq_iff_eq_of_mem {s₁ s₂ : Sphere P} {p : P} (hs₁ : p ∈ s₁) (hs₂ : p ∈ s₂) :
    s₁.center = s₂.center ↔ s₁ = s₂ := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    s₁ s₂ : EuclideanGeometry.Sphere P
    p : P
    hs₁ : Membership.mem s₁ p
    hs₂ : Membership.mem s₂ p
    ⊢ Iff (Eq s₁.center s₂.center) (Eq s₁ s₂)
  -/
  refine ⟨fun h => Sphere.ext h ?_, fun h => h ▸ rfl⟩
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    s₁ s₂ : EuclideanGeometry.Sphere P
    p : P
    hs₁ : Membership.mem s₁ p
    hs₂ : Membership.mem s₂ p
    h : Eq s₁.center s₂.center
    ⊢ Eq s₁.radius s₂.radius
  -/
  rw [mem_sphere] at hs₁ hs₂
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    s₁ s₂ : EuclideanGeometry.Sphere P
    p : P
    hs₁ : Eq (Dist.dist p s₁.center) s₁.radius
    hs₂ : Eq (Dist.dist p s₂.center) s₂.radius
    h : Eq s₁.center s₂.center
    ⊢ Eq s₁.radius s₂.radius
  -/
  rw [← hs₁, ← hs₂, h]
  /-
    🎉 no goals
  -/


theorem Sphere.center_ne_iff_ne_of_mem {s₁ s₂ : Sphere P} {p : P} (hs₁ : p ∈ s₁) (hs₂ : p ∈ s₂) :
    s₁.center ≠ s₂.center ↔ s₁ ≠ s₂ :=
  (Sphere.center_eq_iff_eq_of_mem hs₁ hs₂).not


theorem dist_center_eq_dist_center_of_mem_sphere {p₁ p₂ : P} {s : Sphere P} (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) : dist p₁ s.center = dist p₂ s.center := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    p₁ p₂ : P
    s : EuclideanGeometry.Sphere P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    ⊢ Eq (Dist.dist p₁ s.center) (Dist.dist p₂ s.center)
  -/
  rw [mem_sphere.1 hp₁, mem_sphere.1 hp₂]
  /-
    🎉 no goals
  -/


theorem dist_center_eq_dist_center_of_mem_sphere' {p₁ p₂ : P} {s : Sphere P} (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) : dist s.center p₁ = dist s.center p₂ := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    p₁ p₂ : P
    s : EuclideanGeometry.Sphere P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    ⊢ Eq (Dist.dist s.center p₁) (Dist.dist s.center p₂)
  -/
  rw [mem_sphere'.1 hp₁, mem_sphere'.1 hp₂]
  /-
    🎉 no goals
  -/


/-- A set of points is cospherical if they are equidistant from some
point. In two dimensions, this is the same thing as being
concyclic. -/
def Cospherical (ps : Set P) : Prop :=
  ∃ (center : P) (radius : ℝ), ∀ p ∈ ps, dist p center = radius


/-- The definition of `Cospherical`. -/
theorem cospherical_def (ps : Set P) :
    Cospherical ps ↔ ∃ (center : P) (radius : ℝ), ∀ p ∈ ps, dist p center = radius :=
  Iff.rfl


/-- A set of points is cospherical if and only if they lie in some sphere. -/
theorem cospherical_iff_exists_sphere {ps : Set P} :
    Cospherical ps ↔ ∃ s : Sphere P, ps ⊆ (s : Set P) := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    ps : Set P
    ⊢ Iff (EuclideanGeometry.Cospherical ps) (Exists fun s => HasSubset.Subset ps  …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      P : Type u_2
      inst✝ : MetricSpace P
      ps : Set P
      h : EuclideanGeometry.Cospherical ps
      ⊢ Exists fun s => HasSubset.Subset ps (Metric.sphere s.center s.radius)
    -/
  · rcases h with ⟨c, r, h⟩
    /-
      case refine_1.intro.intro
      P : Type u_2
      inst✝ : MetricSpace P
      ps : Set P
      c : P
      r : Real
      h : ∀ (p : P), Membership.mem ps p → Eq (Dist.dist p c) r
      ⊢ Exists fun s => HasSubset.Subset ps (Metric.sphere s.center s.radius)
    -/
    exact ⟨⟨c, r⟩, h⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : Type u_2
      inst✝ : MetricSpace P
      ps : Set P
      h : Exists fun s => HasSubset.Subset ps (Metric.sphere s.center s.radius)
      ⊢ EuclideanGeometry.Cospherical ps
    -/
  · rcases h with ⟨s, h⟩
    /-
      case refine_2.intro
      P : Type u_2
      inst✝ : MetricSpace P
      ps : Set P
      s : EuclideanGeometry.Sphere P
      h : HasSubset.Subset ps (Metric.sphere s.center s.radius)
      ⊢ EuclideanGeometry.Cospherical ps
    -/
    exact ⟨s.center, s.radius, h⟩
    /-
      🎉 no goals
    -/


/-- The set of points in a sphere is cospherical. -/
theorem Sphere.cospherical (s : Sphere P) : Cospherical (s : Set P) :=
  cospherical_iff_exists_sphere.2 ⟨s, Set.Subset.rfl⟩


/-- A subset of a cospherical set is cospherical. -/
theorem Cospherical.subset {ps₁ ps₂ : Set P} (hs : ps₁ ⊆ ps₂) (hc : Cospherical ps₂) :
    Cospherical ps₁ := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    ps₁ ps₂ : Set P
    hs : HasSubset.Subset ps₁ ps₂
    hc : EuclideanGeometry.Cospherical ps₂
    ⊢ EuclideanGeometry.Cospherical ps₁
  -/
  rcases hc with ⟨c, r, hcr⟩
  /-
    case intro.intro
    P : Type u_2
    inst✝ : MetricSpace P
    ps₁ ps₂ : Set P
    hs : HasSubset.Subset ps₁ ps₂
    c : P
    r : Real
    hcr : ∀ (p : P), Membership.mem ps₂ p → Eq (Dist.dist p c) r
    ⊢ EuclideanGeometry.Cospherical ps₁
  -/
  exact ⟨c, r, fun p hp => hcr p (hs hp)⟩
  /-
    🎉 no goals
  -/


/-- The empty set is cospherical. -/
theorem cospherical_empty [Nonempty P] : Cospherical (∅ : Set P) :=
  let ⟨p⟩ := ‹Nonempty P›
  ⟨p, 0, fun _ => False.elim⟩


/-- A single point is cospherical. -/
theorem cospherical_singleton (p : P) : Cospherical ({p} : Set P) := by
  /-
    P : Type u_2
    inst✝ : MetricSpace P
    p : P
    ⊢ EuclideanGeometry.Cospherical (Singleton.singleton p)
  -/
  use p
  /-
    case h
    P : Type u_2
    inst✝ : MetricSpace P
    p : P
    ⊢ Exists fun radius => ∀ (p_1 : P), Membership.mem (Singleton.singleton p) p_1 …
  -/
  simp
  /-
    🎉 no goals
  -/


include V in
/-- Two points are cospherical. -/
theorem cospherical_pair (p₁ p₂ : P) : Cospherical ({p₁, p₂} : Set P) :=
  ⟨midpoint ℝ p₁ p₂, ‖(2 : ℝ)‖⁻¹ * dist p₁ p₂, by
    /-
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ : P
      ⊢ ∀ (p : P), Membership.mem (Insert.insert p₁ (Singleton.singleton p₂)) p → Eq …
    -/
    rintro p (rfl | rfl | _)
      /-
        case inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₂ p : P
        ⊢ Eq (Dist.dist p (midpoint Real p p₂)) (HMul.hMul (Inv.inv (Norm.norm 2)) (Di …
      -/
    · rw [dist_comm, dist_midpoint_left (𝕜 := ℝ)]
      /-
        🎉 no goals
      -/
      /-
        case inr.refl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ : P
        ⊢ Eq (Dist.dist p₂ (midpoint Real p₁ p₂)) (HMul.hMul (Inv.inv (Norm.norm 2)) ( …
      -/
    · rw [dist_comm, dist_midpoint_right (𝕜 := ℝ)]⟩
      /-
        🎉 no goals
      -/


/-- A set of points is concyclic if it is cospherical and coplanar. (Most results are stated
directly in terms of `Cospherical` instead of using `Concyclic`.) -/
structure Concyclic (ps : Set P) : Prop where
  Cospherical : Cospherical ps
  Coplanar : Coplanar ℝ ps


/-- A subset of a concyclic set is concyclic. -/
theorem Concyclic.subset {ps₁ ps₂ : Set P} (hs : ps₁ ⊆ ps₂) (h : Concyclic ps₂) : Concyclic ps₁ :=
  ⟨h.1.subset hs, h.2.subset hs⟩


/-- The empty set is concyclic. -/
theorem concyclic_empty : Concyclic (∅ : Set P) :=
  ⟨cospherical_empty, coplanar_empty ℝ P⟩


/-- A single point is concyclic. -/
theorem concyclic_singleton (p : P) : Concyclic ({p} : Set P) :=
  ⟨cospherical_singleton p, coplanar_singleton ℝ p⟩


/-- Two points are concyclic. -/
theorem concyclic_pair (p₁ p₂ : P) : Concyclic ({p₁, p₂} : Set P) :=
  ⟨cospherical_pair p₁ p₂, coplanar_pair ℝ p₁ p₂⟩


/-- Any three points in a cospherical set are affinely independent. -/
theorem Cospherical.affineIndependent {s : Set P} (hs : Cospherical s) {p : Fin 3 → P}
    (hps : Set.range p ⊆ s) (hpi : Function.Injective p) : AffineIndependent ℝ p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ⊢ AffineIndependent Real p
  -/
  rw [affineIndependent_iff_not_collinear]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ⊢ Not (Collinear Real (Set.range p))
  -/
  intro hc
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    hc : Collinear Real (Set.range p)
    ⊢ False
  -/
  rw [collinear_iff_of_mem (Set.mem_range_self (0 : Fin 3))] at hc
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    hc : Exists fun v => ∀ (p_1 : P), Membership.mem (Set.range p) p_1 → Exists fu …
    ⊢ False
  -/
  rcases hc with ⟨v, hv⟩
  /-
    case intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv : ∀ (p_1 : P), Membership.mem (Set.range p) p_1 → Exists fun r => Eq p_1 (H …
    ⊢ False
  -/
  rw [Set.forall_mem_range] at hv
  have hv0 : v ≠ 0 := by
    intro h
    have he : p 1 = p 0 := by simpa [h] using hv 1
    exact (by decide : (1 : Fin 3) ≠ 0) (hpi he)
  /-
    case intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv : ∀ (i : Fin 3), Exists fun r => Eq (p i) (HVAdd.hVAdd (HSMul.hSMul r v) (p …
    hv0 : Ne v 0
    ⊢ False
  -/
  rcases hs with ⟨c, r, hs⟩
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv : ∀ (i : Fin 3), Exists fun r => Eq (p i) (HVAdd.hVAdd (HSMul.hSMul r v) (p …
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    ⊢ False
  -/
  have hs' := fun i => hs (p i) (Set.mem_of_mem_of_subset (Set.mem_range_self _) hps)
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv : ∀ (i : Fin 3), Exists fun r => Eq (p i) (HVAdd.hVAdd (HSMul.hSMul r v) (p …
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    hs' : ∀ (i : Fin 3), Eq (Dist.dist (p i) c) r
    ⊢ False
  -/
  choose f hf using hv
  have hsd : ∀ i, dist (f i • v +ᵥ p 0) c = r := by
    intro i
    rw [← hf]
    exact hs' i
  have hf0 : f 0 = 0 := by
    have hf0' := hf 0
    rw [eq_comm, ← @vsub_eq_zero_iff_eq V, vadd_vsub, smul_eq_zero] at hf0'
    simpa [hv0] using hf0'
  have hfi : Function.Injective f := by
    intro i j h
    have hi := hf i
    rw [h, ← hf j] at hi
    exact hpi hi
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    hs' : ∀ (i : Fin 3), Eq (Dist.dist (p i) c) r
    f : Fin 3 → Real
    hf : ∀ (i : Fin 3), Eq (p i) (HVAdd.hVAdd (HSMul.hSMul (f i) v) (p 0))
    hsd : ∀ (i : Fin 3), Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul (f i) v) (p 0)) c …
    hf0 : Eq (f 0) 0
    hfi : Function.Injective f
    ⊢ False
  -/
  simp_rw [← hsd 0, hf0, zero_smul, zero_vadd, dist_smul_vadd_eq_dist (p 0) c hv0] at hsd
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    hs' : ∀ (i : Fin 3), Eq (Dist.dist (p i) c) r
    f : Fin 3 → Real
    hf : ∀ (i : Fin 3), Eq (p i) (HVAdd.hVAdd (HSMul.hSMul (f i) v) (p 0))
    hf0 : Eq (f 0) 0
    hfi : Function.Injective f
    hsd : ∀ (i : Fin 3), Or (Eq (f i) 0) (Eq (f i) (HDiv.hDiv (HMul.hMul (-2) (Inn …
    ⊢ False
  -/
  have hfn0 : ∀ i, i ≠ 0 → f i ≠ 0 := fun i => (hfi.ne_iff' hf0).2
  have hfn0' : ∀ i, i ≠ 0 → f i = -2 * ⟪v, p 0 -ᵥ c⟫ / ⟪v, v⟫ := by
    intro i hi
    have hsdi := hsd i
    simpa [hfn0, hi] using hsdi
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    hs' : ∀ (i : Fin 3), Eq (Dist.dist (p i) c) r
    f : Fin 3 → Real
    hf : ∀ (i : Fin 3), Eq (p i) (HVAdd.hVAdd (HSMul.hSMul (f i) v) (p 0))
    hf0 : Eq (f 0) 0
    hfi : Function.Injective f
    hsd : ∀ (i : Fin 3), Or (Eq (f i) 0) (Eq (f i) (HDiv.hDiv (HMul.hMul (-2) (Inn …
    hfn0 : ∀ (i : Fin 3), Ne i 0 → Ne (f i) 0
    hfn0' : ∀ (i : Fin 3), Ne i 0 → Eq (f i) (HDiv.hDiv (HMul.hMul (-2) (Inner.inn …
    ⊢ False
  -/
  have hf12 : f 1 = f 2 := by rw [hfn0' 1 (by decide), hfn0' 2 (by decide)]
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    v : V
    hv0 : Ne v 0
    c : P
    r : Real
    hs : ∀ (p : P), Membership.mem s p → Eq (Dist.dist p c) r
    hs' : ∀ (i : Fin 3), Eq (Dist.dist (p i) c) r
    f : Fin 3 → Real
    hf : ∀ (i : Fin 3), Eq (p i) (HVAdd.hVAdd (HSMul.hSMul (f i) v) (p 0))
    hf0 : Eq (f 0) 0
    hfi : Function.Injective f
    hsd : ∀ (i : Fin 3), Or (Eq (f i) 0) (Eq (f i) (HDiv.hDiv (HMul.hMul (-2) (Inn …
    hfn0 : ∀ (i : Fin 3), Ne i 0 → Ne (f i) 0
    hfn0' : ∀ (i : Fin 3), Ne i 0 → Eq (f i) (HDiv.hDiv (HMul.hMul (-2) (Inner.inn …
    hf12 : Eq (f 1) (f 2)
    ⊢ False
  -/
  exact (by decide : (1 : Fin 3) ≠ 2) (hfi hf12)
  /-
    🎉 no goals
  -/


/-- Any three points in a cospherical set are affinely independent. -/
theorem Cospherical.affineIndependent_of_mem_of_ne {s : Set P} (hs : Cospherical s) {p₁ p₂ p₃ : P}
    (h₁ : p₁ ∈ s) (h₂ : p₂ ∈ s) (h₃ : p₃ ∈ s) (h₁₂ : p₁ ≠ p₂) (h₁₃ : p₁ ≠ p₃) (h₂₃ : p₂ ≠ p₃) :
    AffineIndependent ℝ ![p₁, p₂, p₃] := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    hs : EuclideanGeometry.Cospherical s
    p₁ p₂ p₃ : P
    h₁ : Membership.mem s p₁
    h₂ : Membership.mem s p₂
    h₃ : Membership.mem s p₃
    h₁₂ : Ne p₁ p₂
    h₁₃ : Ne p₁ p₃
    h₂₃ : Ne p₂ p₃
    ⊢ AffineIndependent Real (Matrix.vecCons p₁ (Matrix.vecCons p₂ (Matrix.vecCons …
  -/
  refine hs.affineIndependent ?_ ?_
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      hs : EuclideanGeometry.Cospherical s
      p₁ p₂ p₃ : P
      h₁ : Membership.mem s p₁
      h₂ : Membership.mem s p₂
      h₃ : Membership.mem s p₃
      h₁₂ : Ne p₁ p₂
      h₁₃ : Ne p₁ p₃
      h₂₃ : Ne p₂ p₃
      ⊢ HasSubset.Subset (Set.range (Matrix.vecCons p₁ (Matrix.vecCons p₂ (Matrix.ve …
    -/
  · simp [h₁, h₂, h₃, Set.insert_subset_iff]
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
      s : Set P
      hs : EuclideanGeometry.Cospherical s
      p₁ p₂ p₃ : P
      h₁ : Membership.mem s p₁
      h₂ : Membership.mem s p₂
      h₃ : Membership.mem s p₃
      h₁₂ : Ne p₁ p₂
      h₁₃ : Ne p₁ p₃
      h₂₃ : Ne p₂ p₃
      ⊢ Function.Injective (Matrix.vecCons p₁ (Matrix.vecCons p₂ (Matrix.vecCons p₃  …
    -/
  · erw [Fin.cons_injective_iff, Fin.cons_injective_iff]
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      hs : EuclideanGeometry.Cospherical s
      p₁ p₂ p₃ : P
      h₁ : Membership.mem s p₁
      h₂ : Membership.mem s p₂
      h₃ : Membership.mem s p₃
      h₁₂ : Ne p₁ p₂
      h₁₃ : Ne p₁ p₃
      h₂₃ : Ne p₂ p₃
      ⊢ And (Not (Membership.mem (Set.range (Matrix.vecCons p₂ (Matrix.vecCons p₃ Ma …
    -/
    simp [h₁₂, h₁₃, h₂₃, Function.Injective, eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


/-- The three points of a cospherical set are affinely independent. -/
theorem Cospherical.affineIndependent_of_ne {p₁ p₂ p₃ : P} (hs : Cospherical ({p₁, p₂, p₃} : Set P))
    (h₁₂ : p₁ ≠ p₂) (h₁₃ : p₁ ≠ p₃) (h₂₃ : p₂ ≠ p₃) : AffineIndependent ℝ ![p₁, p₂, p₃] :=
  hs.affineIndependent_of_mem_of_ne (Set.mem_insert _ _)
    (Set.mem_insert_of_mem _ (Set.mem_insert _ _))
    (Set.mem_insert_of_mem _ (Set.mem_insert_of_mem _ (Set.mem_singleton _))) h₁₂ h₁₃ h₂₃


/-- Suppose that `p₁` and `p₂` lie in spheres `s₁` and `s₂`. Then the vector between the centers
of those spheres is orthogonal to that between `p₁` and `p₂`; this is a version of
`inner_vsub_vsub_of_dist_eq_of_dist_eq` for bundled spheres. (In two dimensions, this says that
the diagonals of a kite are orthogonal.) -/
theorem inner_vsub_vsub_of_mem_sphere_of_mem_sphere {p₁ p₂ : P} {s₁ s₂ : Sphere P} (hp₁s₁ : p₁ ∈ s₁)
    (hp₂s₁ : p₂ ∈ s₁) (hp₁s₂ : p₁ ∈ s₂) (hp₂s₂ : p₂ ∈ s₂) :
    ⟪s₂.center -ᵥ s₁.center, p₂ -ᵥ p₁⟫ = 0 :=
  inner_vsub_vsub_of_dist_eq_of_dist_eq (dist_center_eq_dist_center_of_mem_sphere hp₁s₁ hp₂s₁)
    (dist_center_eq_dist_center_of_mem_sphere hp₁s₂ hp₂s₂)


/-- Two spheres intersect in at most two points in a two-dimensional subspace containing their
centers; this is a version of `eq_of_dist_eq_of_dist_eq_of_mem_of_finrank_eq_two` for bundled
spheres. -/
theorem eq_of_mem_sphere_of_mem_sphere_of_mem_of_finrank_eq_two {s : AffineSubspace ℝ P}
    [FiniteDimensional ℝ s.direction] (hd : finrank ℝ s.direction = 2) {s₁ s₂ : Sphere P}
    {p₁ p₂ p : P} (hs₁ : s₁.center ∈ s) (hs₂ : s₂.center ∈ s) (hp₁s : p₁ ∈ s) (hp₂s : p₂ ∈ s)
    (hps : p ∈ s) (hs : s₁ ≠ s₂) (hp : p₁ ≠ p₂) (hp₁s₁ : p₁ ∈ s₁) (hp₂s₁ : p₂ ∈ s₁) (hps₁ : p ∈ s₁)
    (hp₁s₂ : p₁ ∈ s₂) (hp₂s₂ : p₂ ∈ s₂) (hps₂ : p ∈ s₂) : p = p₁ ∨ p = p₂ :=
  eq_of_dist_eq_of_dist_eq_of_mem_of_finrank_eq_two hd hs₁ hs₂ hp₁s hp₂s hps
    ((Sphere.center_ne_iff_ne_of_mem hps₁ hps₂).2 hs) hp hp₁s₁ hp₂s₁ hps₁ hp₁s₂ hp₂s₂ hps₂


/-- Two spheres intersect in at most two points in two-dimensional space; this is a version of
`eq_of_dist_eq_of_dist_eq_of_finrank_eq_two` for bundled spheres. -/
theorem eq_of_mem_sphere_of_mem_sphere_of_finrank_eq_two [FiniteDimensional ℝ V]
    (hd : finrank ℝ V = 2) {s₁ s₂ : Sphere P} {p₁ p₂ p : P} (hs : s₁ ≠ s₂) (hp : p₁ ≠ p₂)
    (hp₁s₁ : p₁ ∈ s₁) (hp₂s₁ : p₂ ∈ s₁) (hps₁ : p ∈ s₁) (hp₁s₂ : p₁ ∈ s₂) (hp₂s₂ : p₂ ∈ s₂)
    (hps₂ : p ∈ s₂) : p = p₁ ∨ p = p₂ :=
  eq_of_dist_eq_of_dist_eq_of_finrank_eq_two hd ((Sphere.center_ne_iff_ne_of_mem hps₁ hps₂).2 hs) hp
    hp₁s₁ hp₂s₁ hps₁ hp₁s₂ hp₂s₂ hps₂


/-- Given a point on a sphere and a point not outside it, the inner product between the
difference of those points and the radius vector is positive unless the points are equal. -/
theorem inner_pos_or_eq_of_dist_le_radius {s : Sphere P} {p₁ p₂ : P} (hp₁ : p₁ ∈ s)
    (hp₂ : dist p₂ s.center ≤ s.radius) : 0 < ⟪p₁ -ᵥ p₂, p₁ -ᵥ s.center⟫ ∨ p₁ = p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
    ⊢ Or (LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))) (Eq p₁  …
  -/
  by_cases h : p₁ = p₂; · exact Or.inr h
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
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
    h : Not (Eq p₁ p₂)
    ⊢ Or (LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))) (Eq p₁  …
  -/
  refine Or.inl ?_
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
    h : Not (Eq p₁ p₂)
    ⊢ LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
  -/
  rw [mem_sphere] at hp₁
  rw [← vsub_sub_vsub_cancel_right p₁ p₂ s.center, inner_sub_left,
    real_inner_self_eq_norm_mul_norm, sub_pos]
  refine lt_of_le_of_ne
    ((real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))) ?_
    /-
      case neg.refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ p₂ : P
      hp₁ : Eq (Dist.dist p₁ s.center) s.radius
      hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
      h : Not (Eq p₁ p₂)
      ⊢ LE.le (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub p₁ s.center))
    -/
  · rwa [← dist_eq_norm_vsub, ← dist_eq_norm_vsub, hp₁]
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ p₂ : P
      hp₁ : Eq (Dist.dist p₁ s.center) s.radius
      hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
      h : Not (Eq p₁ p₂)
      ⊢ Ne (Inner.inner (VSub.vsub p₂ s.center) (VSub.vsub p₁ s.center)) (HMul.hMul  …
    -/
  · rcases hp₂.lt_or_eq with (hp₂' | hp₂')
      /-
        case neg.refine_2.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₁ p₂ : P
        hp₁ : Eq (Dist.dist p₁ s.center) s.radius
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        h : Not (Eq p₁ p₂)
        hp₂' : LT.lt (Dist.dist p₂ s.center) s.radius
        ⊢ Ne (Inner.inner (VSub.vsub p₂ s.center) (VSub.vsub p₁ s.center)) (HMul.hMul  …
      -/
    · refine ((real_inner_le_norm _ _).trans_lt (mul_lt_mul_of_pos_right ?_ ?_)).ne
        /-
          case neg.refine_2.inl.refine_1
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₁ p₂ : P
          hp₁ : Eq (Dist.dist p₁ s.center) s.radius
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          h : Not (Eq p₁ p₂)
          hp₂' : LT.lt (Dist.dist p₂ s.center) s.radius
          ⊢ LT.lt (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub p₁ s.center))
        -/
      · rwa [← hp₁, @dist_eq_norm_vsub V, @dist_eq_norm_vsub V] at hp₂'
        /-
          🎉 no goals
        -/
        /-
          case neg.refine_2.inl.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₁ p₂ : P
          hp₁ : Eq (Dist.dist p₁ s.center) s.radius
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          h : Not (Eq p₁ p₂)
          hp₂' : LT.lt (Dist.dist p₂ s.center) s.radius
          ⊢ LT.lt 0 (Norm.norm (VSub.vsub p₁ s.center))
        -/
      · rw [norm_pos_iff, vsub_ne_zero]
        /-
          case neg.refine_2.inl.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₁ p₂ : P
          hp₁ : Eq (Dist.dist p₁ s.center) s.radius
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          h : Not (Eq p₁ p₂)
          hp₂' : LT.lt (Dist.dist p₂ s.center) s.radius
          ⊢ Ne p₁ s.center
        -/
        rintro rfl
        /-
          case neg.refine_2.inl.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₂ : P
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          hp₂' : LT.lt (Dist.dist p₂ s.center) s.radius
          hp₁ : Eq (Dist.dist s.center s.center) s.radius
          h : Not (Eq s.center p₂)
          ⊢ False
        -/
        rw [← hp₁] at hp₂'
        /-
          case neg.refine_2.inl.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₂ : P
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          hp₂' : LT.lt (Dist.dist p₂ s.center) (Dist.dist s.center s.center)
          hp₁ : Eq (Dist.dist s.center s.center) s.radius
          h : Not (Eq s.center p₂)
          ⊢ False
        -/
        refine (dist_nonneg.not_lt : ¬dist p₂ s.center < 0) ?_
        /-
          case neg.refine_2.inl.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          s : EuclideanGeometry.Sphere P
          p₂ : P
          hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
          hp₂' : LT.lt (Dist.dist p₂ s.center) (Dist.dist s.center s.center)
          hp₁ : Eq (Dist.dist s.center s.center) s.radius
          h : Not (Eq s.center p₂)
          ⊢ LT.lt (Dist.dist p₂ s.center) 0
        -/
        simpa using hp₂'
        /-
          🎉 no goals
        -/
      /-
        case neg.refine_2.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₁ p₂ : P
        hp₁ : Eq (Dist.dist p₁ s.center) s.radius
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        h : Not (Eq p₁ p₂)
        hp₂' : Eq (Dist.dist p₂ s.center) s.radius
        ⊢ Ne (Inner.inner (VSub.vsub p₂ s.center) (VSub.vsub p₁ s.center)) (HMul.hMul  …
      -/
    · rw [← hp₁, @dist_eq_norm_vsub V, @dist_eq_norm_vsub V] at hp₂'
      /-
        case neg.refine_2.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₁ p₂ : P
        hp₁ : Eq (Dist.dist p₁ s.center) s.radius
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        h : Not (Eq p₁ p₂)
        hp₂' : Eq (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub p₁ s.cente …
        ⊢ Ne (Inner.inner (VSub.vsub p₂ s.center) (VSub.vsub p₁ s.center)) (HMul.hMul  …
      -/
      nth_rw 1 [← hp₂']
      rw [Ne, inner_eq_norm_mul_iff_real, hp₂', ← sub_eq_zero, ← smul_sub,
        vsub_sub_vsub_cancel_right, ← Ne, smul_ne_zero_iff, vsub_ne_zero,
        and_iff_left (Ne.symm h), norm_ne_zero_iff, vsub_ne_zero]
      /-
        case neg.refine_2.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₁ p₂ : P
        hp₁ : Eq (Dist.dist p₁ s.center) s.radius
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        h : Not (Eq p₁ p₂)
        hp₂' : Eq (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub p₁ s.cente …
        ⊢ Ne p₁ s.center
      -/
      rintro rfl
      /-
        case neg.refine_2.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₂ : P
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        hp₁ : Eq (Dist.dist s.center s.center) s.radius
        h : Not (Eq s.center p₂)
        hp₂' : Eq (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub s.center s …
        ⊢ False
      -/
      refine h (Eq.symm ?_)
      /-
        case neg.refine_2.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : EuclideanGeometry.Sphere P
        p₂ : P
        hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
        hp₁ : Eq (Dist.dist s.center s.center) s.radius
        h : Not (Eq s.center p₂)
        hp₂' : Eq (Norm.norm (VSub.vsub p₂ s.center)) (Norm.norm (VSub.vsub s.center s …
        ⊢ Eq p₂ s.center
      -/
      simpa using hp₂'
      /-
        🎉 no goals
      -/


/-- Given a point on a sphere and a point not outside it, the inner product between the
difference of those points and the radius vector is nonnegative. -/
theorem inner_nonneg_of_dist_le_radius {s : Sphere P} {p₁ p₂ : P} (hp₁ : p₁ ∈ s)
    (hp₂ : dist p₂ s.center ≤ s.radius) : 0 ≤ ⟪p₁ -ᵥ p₂, p₁ -ᵥ s.center⟫ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
    ⊢ LE.le 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
  -/
  rcases inner_pos_or_eq_of_dist_le_radius hp₁ hp₂ with (h | rfl)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ p₂ : P
      hp₁ : Membership.mem s p₁
      hp₂ : LE.le (Dist.dist p₂ s.center) s.radius
      h : LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
      ⊢ LE.le 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
    -/
  · exact h.le
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ : P
      hp₁ : Membership.mem s p₁
      hp₂ : LE.le (Dist.dist p₁ s.center) s.radius
      ⊢ LE.le 0 (Inner.inner (VSub.vsub p₁ p₁) (VSub.vsub p₁ s.center))
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- Given a point on a sphere and a point inside it, the inner product between the difference of
those points and the radius vector is positive. -/
theorem inner_pos_of_dist_lt_radius {s : Sphere P} {p₁ p₂ : P} (hp₁ : p₁ ∈ s)
    (hp₂ : dist p₂ s.center < s.radius) : 0 < ⟪p₁ -ᵥ p₂, p₁ -ᵥ s.center⟫ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : EuclideanGeometry.Sphere P
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LT.lt (Dist.dist p₂ s.center) s.radius
    ⊢ LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
  -/
  by_cases h : p₁ = p₂
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ p₂ : P
      hp₁ : Membership.mem s p₁
      hp₂ : LT.lt (Dist.dist p₂ s.center) s.radius
      h : Eq p₁ p₂
      ⊢ LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
    -/
  · rw [h, mem_sphere] at hp₁
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : EuclideanGeometry.Sphere P
      p₁ p₂ : P
      hp₁ : Eq (Dist.dist p₂ s.center) s.radius
      hp₂ : LT.lt (Dist.dist p₂ s.center) s.radius
      h : Eq p₁ p₂
      ⊢ LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
    -/
    exact False.elim (hp₂.ne hp₁)
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
    p₁ p₂ : P
    hp₁ : Membership.mem s p₁
    hp₂ : LT.lt (Dist.dist p₂ s.center) s.radius
    h : Not (Eq p₁ p₂)
    ⊢ LT.lt 0 (Inner.inner (VSub.vsub p₁ p₂) (VSub.vsub p₁ s.center))
  -/
  exact (inner_pos_or_eq_of_dist_le_radius hp₁ hp₂.le).resolve_right h
  /-
    🎉 no goals
  -/


/-- Given three collinear points, two on a sphere and one not outside it, the one not outside it
is weakly between the other two points. -/
theorem wbtw_of_collinear_of_dist_center_le_radius {s : Sphere P} {p₁ p₂ p₃ : P}
    (h : Collinear ℝ ({p₁, p₂, p₃} : Set P)) (hp₁ : p₁ ∈ s) (hp₂ : dist p₂ s.center ≤ s.radius)
    (hp₃ : p₃ ∈ s) (hp₁p₃ : p₁ ≠ p₃) : Wbtw ℝ p₁ p₂ p₃ :=
  h.wbtw_of_dist_eq_of_dist_le hp₁ hp₂ hp₃ hp₁p₃


/-- Given three collinear points, two on a sphere and one inside it, the one inside it is
strictly between the other two points. -/
theorem sbtw_of_collinear_of_dist_center_lt_radius {s : Sphere P} {p₁ p₂ p₃ : P}
    (h : Collinear ℝ ({p₁, p₂, p₃} : Set P)) (hp₁ : p₁ ∈ s) (hp₂ : dist p₂ s.center < s.radius)
    (hp₃ : p₃ ∈ s) (hp₁p₃ : p₁ ≠ p₃) : Sbtw ℝ p₁ p₂ p₃ :=
  h.sbtw_of_dist_eq_of_dist_lt hp₁ hp₂ hp₃ hp₁p₃


