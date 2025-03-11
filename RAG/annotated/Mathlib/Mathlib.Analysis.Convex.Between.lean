/-- The segment of points weakly between `x` and `y`. When convexity is refactored to support
abstract affine combination spaces, this will no longer need to be a separate definition from
`segment`. However, lemmas involving `+ᵥ` or `-ᵥ` will still be relevant after such a
refactoring, as distinct from versions involving `+` or `-` in a module. -/
def affineSegment (x y : P) :=
  lineMap x y '' Set.Icc (0 : R) 1


theorem affineSegment_eq_segment (x y : V) : affineSegment R x y = segment R x y := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : OrderedRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    x y : V
    ⊢ Eq (affineSegment R x y) (segment R x y)
  -/
  rw [segment_eq_image_lineMap, affineSegment]
  /-
    🎉 no goals
  -/


theorem affineSegment_comm (x y : P) : affineSegment R x y = affineSegment R y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq (affineSegment R x y) (affineSegment R y x)
  -/
  refine Set.ext fun z => ?_
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Membership.mem (affineSegment R x y) z) (Membership.mem (affineSegment  …
  -/
  constructor <;>
      /-
        case mp
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : OrderedRing R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y z : P
        ⊢ Membership.mem (affineSegment R x y) z → Membership.mem (affineSegment R y x …
      -/
      /-
        case mp.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : OrderedRing R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y z : P
        t : R
        ht : Membership.mem (Set.Icc 0 1) t
        hxy : Eq ((AffineMap.lineMap x y) t) z
        ⊢ Membership.mem (affineSegment R y x) z
      -/
        /-
          case mp.intro.intro.refine_1
          R : Type u_1
          V : Type u_2
          P : Type u_4
          inst✝³ : OrderedRing R
          inst✝² : AddCommGroup V
          inst✝¹ : Module R V
          inst✝ : AddTorsor V P
          x y z : P
          t : R
          ht : Membership.mem (Set.Icc 0 1) t
          hxy : Eq ((AffineMap.lineMap x y) t) z
          ⊢ Membership.mem (Set.Icc 0 1) (HSub.hSub 1 t)
        -/
        /-
          🎉 no goals
        -/
        /-
          case mp.intro.intro.refine_2
          R : Type u_1
          V : Type u_2
          P : Type u_4
          inst✝³ : OrderedRing R
          inst✝² : AddCommGroup V
          inst✝¹ : Module R V
          inst✝ : AddTorsor V P
          x y z : P
          t : R
          ht : Membership.mem (Set.Icc 0 1) t
          hxy : Eq ((AffineMap.lineMap x y) t) z
          ⊢ Eq ((AffineMap.lineMap y x) (HSub.hSub 1 t)) z
        -/
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.refine_1
          R : Type u_1
          V : Type u_2
          P : Type u_4
          inst✝³ : OrderedRing R
          inst✝² : AddCommGroup V
          inst✝¹ : Module R V
          inst✝ : AddTorsor V P
          x y z : P
          t : R
          ht : Membership.mem (Set.Icc 0 1) t
          hxy : Eq ((AffineMap.lineMap y x) t) z
          ⊢ Membership.mem (Set.Icc 0 1) (HSub.hSub 1 t)
        -/
      · rwa [Set.sub_mem_Icc_iff_right, sub_self, sub_zero]
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.refine_2
          R : Type u_1
          V : Type u_2
          P : Type u_4
          inst✝³ : OrderedRing R
          inst✝² : AddCommGroup V
          inst✝¹ : Module R V
          inst✝ : AddTorsor V P
          x y z : P
          t : R
          ht : Membership.mem (Set.Icc 0 1) t
          hxy : Eq ((AffineMap.lineMap y x) t) z
          ⊢ Eq ((AffineMap.lineMap x y) (HSub.hSub 1 t)) z
        -/
      · rwa [lineMap_apply_one_sub]
        /-
          🎉 no goals
        -/


theorem left_mem_affineSegment (x y : P) : x ∈ affineSegment R x y :=
  ⟨0, Set.left_mem_Icc.2 zero_le_one, lineMap_apply_zero _ _⟩


theorem right_mem_affineSegment (x y : P) : y ∈ affineSegment R x y :=
  ⟨1, Set.right_mem_Icc.2 zero_le_one, lineMap_apply_one _ _⟩


@[simp]
theorem affineSegment_same (x : P) : affineSegment R x x = {x} := by
  simp_rw [affineSegment, lineMap_same, AffineMap.coe_const, Function.const,
    (Set.nonempty_Icc.mpr zero_le_one).image_const]


@[simp]
theorem affineSegment_image (f : P →ᵃ[R] P') (x y : P) :
    f '' affineSegment R x y = affineSegment R (f x) (f y) := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    f : AffineMap R P P'
    x y : P
    ⊢ Eq (Set.image (⇑f) (affineSegment R x y)) (affineSegment R (f x) (f y))
  -/
  rw [affineSegment, affineSegment, Set.image_image, ← comp_lineMap]
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    f : AffineMap R P P'
    x y : P
    ⊢ Eq (Set.image (fun x_1 => f ((AffineMap.lineMap x y) x_1)) (Set.Icc 0 1)) (S …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem affineSegment_const_vadd_image (x y : P) (v : V) :
    (v +ᵥ ·) '' affineSegment R x y = affineSegment R (v +ᵥ x) (v +ᵥ y) :=
  affineSegment_image (AffineEquiv.constVAdd R P v : P →ᵃ[R] P) x y


@[simp]
theorem affineSegment_vadd_const_image (x y : V) (p : P) :
    (· +ᵥ p) '' affineSegment R x y = affineSegment R (x +ᵥ p) (y +ᵥ p) :=
  affineSegment_image (AffineEquiv.vaddConst R p : V →ᵃ[R] P) x y


@[simp]
theorem affineSegment_const_vsub_image (x y p : P) :
    (p -ᵥ ·) '' affineSegment R x y = affineSegment R (p -ᵥ x) (p -ᵥ y) :=
  affineSegment_image (AffineEquiv.constVSub R p : P →ᵃ[R] V) x y


@[simp]
theorem affineSegment_vsub_const_image (x y p : P) :
    (· -ᵥ p) '' affineSegment R x y = affineSegment R (x -ᵥ p) (y -ᵥ p) :=
  affineSegment_image ((AffineEquiv.vaddConst R p).symm : P →ᵃ[R] V) x y


@[simp]
theorem mem_const_vadd_affineSegment {x y z : P} (v : V) :
    v +ᵥ z ∈ affineSegment R (v +ᵥ x) (v +ᵥ y) ↔ z ∈ affineSegment R x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    v : V
    ⊢ Iff (Membership.mem (affineSegment R (HVAdd.hVAdd v x) (HVAdd.hVAdd v y)) (H …
  -/
  rw [← affineSegment_const_vadd_image, (AddAction.injective v).mem_set_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_vadd_const_affineSegment {x y z : V} (p : P) :
    z +ᵥ p ∈ affineSegment R (x +ᵥ p) (y +ᵥ p) ↔ z ∈ affineSegment R x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : V
    p : P
    ⊢ Iff (Membership.mem (affineSegment R (HVAdd.hVAdd x p) (HVAdd.hVAdd y p)) (H …
  -/
  rw [← affineSegment_vadd_const_image, (vadd_right_injective p).mem_set_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_const_vsub_affineSegment {x y z : P} (p : P) :
    p -ᵥ z ∈ affineSegment R (p -ᵥ x) (p -ᵥ y) ↔ z ∈ affineSegment R x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z p : P
    ⊢ Iff (Membership.mem (affineSegment R (VSub.vsub p x) (VSub.vsub p y)) (VSub. …
  -/
  rw [← affineSegment_const_vsub_image, (vsub_right_injective p).mem_set_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_vsub_const_affineSegment {x y z : P} (p : P) :
    z -ᵥ p ∈ affineSegment R (x -ᵥ p) (y -ᵥ p) ↔ z ∈ affineSegment R x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z p : P
    ⊢ Iff (Membership.mem (affineSegment R (VSub.vsub x p) (VSub.vsub y p)) (VSub. …
  -/
  rw [← affineSegment_vsub_const_image, (vsub_left_injective p).mem_set_image]
  /-
    🎉 no goals
  -/


/-- The point `y` is weakly between `x` and `z`. -/
def Wbtw (x y z : P) : Prop :=
  y ∈ affineSegment R x z


/-- The point `y` is strictly between `x` and `z`. -/
def Sbtw (x y z : P) : Prop :=
  Wbtw R x y z ∧ y ≠ x ∧ y ≠ z


lemma mem_segment_iff_wbtw {x y z : V} : y ∈ segment R x z ↔ Wbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    inst✝² : OrderedRing R
    inst✝¹ : AddCommGroup V
    inst✝ : Module R V
    x y z : V
    ⊢ Iff (Membership.mem (segment R x z) y) (Wbtw R x y z)
  -/
  rw [Wbtw, affineSegment_eq_segment]
  /-
    🎉 no goals
  -/


alias ⟨_, Wbtw.mem_segment⟩ := mem_segment_iff_wbtw


lemma Convex.mem_of_wbtw {p₀ p₁ p₂ : V} {s : Set V} (hs : Convex R s) (h₀₁₂ : Wbtw R p₀ p₁ p₂)
    (h₀ : p₀ ∈ s) (h₂ : p₂ ∈ s) : p₁ ∈ s := hs.segment_subset h₀ h₂ h₀₁₂.mem_segment


lemma AffineSubspace.mem_of_wbtw {s : AffineSubspace R P} {x y z : P} (hxyz : Wbtw R x y z)
                                            /-
                                              R : Type u_1
                                              V : Type u_2
                                              P : Type u_4
                                              inst✝³ : OrderedRing R
                                              inst✝² : AddCommGroup V
                                              inst✝¹ : Module R V
                                              inst✝ : AddTorsor V P
                                              s : AffineSubspace R P
                                              x y z : P
                                              hxyz : Wbtw R x y z
                                              hx : Membership.mem s x
                                              hz : Membership.mem s z
                                              ⊢ Membership.mem s y
                                            -/
    (hx : x ∈ s) (hz : z ∈ s) : y ∈ s := by obtain ⟨ε, -, rfl⟩ := hxyz; exact lineMap_mem _ hx hz
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem Wbtw.map {x y z : P} (h : Wbtw R x y z) (f : P →ᵃ[R] P') : Wbtw R (f x) (f y) (f z) := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    h : Wbtw R x y z
    f : AffineMap R P P'
    ⊢ Wbtw R (f x) (f y) (f z)
  -/
  rw [Wbtw, ← affineSegment_image]
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    h : Wbtw R x y z
    f : AffineMap R P P'
    ⊢ Membership.mem (Set.image (⇑f) (affineSegment R x z)) (f y)
  -/
  exact Set.mem_image_of_mem _ h
  /-
    🎉 no goals
  -/


theorem Function.Injective.wbtw_map_iff {x y z : P} {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    Wbtw R (f x) (f y) (f z) ↔ Wbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff (Wbtw R (f x) (f y) (f z)) (Wbtw R x y z)
  -/
  refine ⟨fun h => ?_, fun h => h.map _⟩
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    h : Wbtw R (f x) (f y) (f z)
    ⊢ Wbtw R x y z
  -/
  rwa [Wbtw, ← affineSegment_image, hf.mem_set_image] at h
  /-
    🎉 no goals
  -/


theorem Function.Injective.sbtw_map_iff {x y z : P} {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    Sbtw R (f x) (f y) (f z) ↔ Sbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff (Sbtw R (f x) (f y) (f z)) (Sbtw R x y z)
  -/
  simp_rw [Sbtw, hf.wbtw_map_iff, hf.ne_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem AffineEquiv.wbtw_map_iff {x y z : P} (f : P ≃ᵃ[R] P') :
    Wbtw R (f x) (f y) (f z) ↔ Wbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineEquiv R P P'
    ⊢ Iff (Wbtw R (f x) (f y) (f z)) (Wbtw R x y z)
  -/
  have : Function.Injective f.toAffineMap := f.injective
  -- `refine` or `exact` are very slow, `apply` is fast. Please check before golfing.
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineEquiv R P P'
    this : Function.Injective ⇑↑f
    ⊢ Iff (Wbtw R (f x) (f y) (f z)) (Wbtw R x y z)
  -/
  apply this.wbtw_map_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem AffineEquiv.sbtw_map_iff {x y z : P} (f : P ≃ᵃ[R] P') :
    Sbtw R (f x) (f y) (f z) ↔ Sbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineEquiv R P P'
    ⊢ Iff (Sbtw R (f x) (f y) (f z)) (Sbtw R x y z)
  -/
  have : Function.Injective f.toAffineMap := f.injective
  -- `refine` or `exact` are very slow, `apply` is fast. Please check before golfing.
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : OrderedRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    x y z : P
    f : AffineEquiv R P P'
    this : Function.Injective ⇑↑f
    ⊢ Iff (Sbtw R (f x) (f y) (f z)) (Sbtw R x y z)
  -/
  apply this.sbtw_map_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem wbtw_const_vadd_iff {x y z : P} (v : V) :
    Wbtw R (v +ᵥ x) (v +ᵥ y) (v +ᵥ z) ↔ Wbtw R x y z :=
  mem_const_vadd_affineSegment _


@[simp]
theorem wbtw_vadd_const_iff {x y z : V} (p : P) :
    Wbtw R (x +ᵥ p) (y +ᵥ p) (z +ᵥ p) ↔ Wbtw R x y z :=
  mem_vadd_const_affineSegment _


@[simp]
theorem wbtw_const_vsub_iff {x y z : P} (p : P) :
    Wbtw R (p -ᵥ x) (p -ᵥ y) (p -ᵥ z) ↔ Wbtw R x y z :=
  mem_const_vsub_affineSegment _


@[simp]
theorem wbtw_vsub_const_iff {x y z : P} (p : P) :
    Wbtw R (x -ᵥ p) (y -ᵥ p) (z -ᵥ p) ↔ Wbtw R x y z :=
  mem_vsub_const_affineSegment _


@[simp]
theorem sbtw_const_vadd_iff {x y z : P} (v : V) :
    Sbtw R (v +ᵥ x) (v +ᵥ y) (v +ᵥ z) ↔ Sbtw R x y z := by
  rw [Sbtw, Sbtw, wbtw_const_vadd_iff, (AddAction.injective v).ne_iff,
    (AddAction.injective v).ne_iff]


@[simp]
theorem sbtw_vadd_const_iff {x y z : V} (p : P) :
    Sbtw R (x +ᵥ p) (y +ᵥ p) (z +ᵥ p) ↔ Sbtw R x y z := by
  rw [Sbtw, Sbtw, wbtw_vadd_const_iff, (vadd_right_injective p).ne_iff,
    (vadd_right_injective p).ne_iff]


@[simp]
theorem sbtw_const_vsub_iff {x y z : P} (p : P) :
    Sbtw R (p -ᵥ x) (p -ᵥ y) (p -ᵥ z) ↔ Sbtw R x y z := by
  rw [Sbtw, Sbtw, wbtw_const_vsub_iff, (vsub_right_injective p).ne_iff,
    (vsub_right_injective p).ne_iff]


@[simp]
theorem sbtw_vsub_const_iff {x y z : P} (p : P) :
    Sbtw R (x -ᵥ p) (y -ᵥ p) (z -ᵥ p) ↔ Sbtw R x y z := by
  rw [Sbtw, Sbtw, wbtw_vsub_const_iff, (vsub_left_injective p).ne_iff,
    (vsub_left_injective p).ne_iff]


theorem Sbtw.wbtw {x y z : P} (h : Sbtw R x y z) : Wbtw R x y z :=
  h.1


theorem Sbtw.ne_left {x y z : P} (h : Sbtw R x y z) : y ≠ x :=
  h.2.1


theorem Sbtw.left_ne {x y z : P} (h : Sbtw R x y z) : x ≠ y :=
  h.2.1.symm


theorem Sbtw.ne_right {x y z : P} (h : Sbtw R x y z) : y ≠ z :=
  h.2.2


theorem Sbtw.right_ne {x y z : P} (h : Sbtw R x y z) : z ≠ y :=
  h.2.2.symm


theorem Sbtw.mem_image_Ioo {x y z : P} (h : Sbtw R x y z) :
    y ∈ lineMap x z '' Set.Ioo (0 : R) 1 := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Sbtw R x y z
    ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) y
  -/
  rcases h with ⟨⟨t, ht, rfl⟩, hyx, hyz⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht : Membership.mem (Set.Icc 0 1) t
    hyx : Ne ((AffineMap.lineMap x z) t) x
    hyz : Ne ((AffineMap.lineMap x z) t) z
    ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) ((Affine …
  -/
  rcases Set.eq_endpoints_or_mem_Ioo_of_mem_Icc ht with (rfl | rfl | ho)
    /-
      case intro.intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      ht : Membership.mem (Set.Icc 0 1) 0
      hyx : Ne ((AffineMap.lineMap x z) 0) x
      hyz : Ne ((AffineMap.lineMap x z) 0) z
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) ((Affine …
    -/
  · exfalso
    /-
      case intro.intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      ht : Membership.mem (Set.Icc 0 1) 0
      hyx : Ne ((AffineMap.lineMap x z) 0) x
      hyz : Ne ((AffineMap.lineMap x z) 0) z
      ⊢ False
    -/
    exact hyx (lineMap_apply_zero _ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      ht : Membership.mem (Set.Icc 0 1) 1
      hyx : Ne ((AffineMap.lineMap x z) 1) x
      hyz : Ne ((AffineMap.lineMap x z) 1) z
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) ((Affine …
    -/
  · exfalso
    /-
      case intro.intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      ht : Membership.mem (Set.Icc 0 1) 1
      hyx : Ne ((AffineMap.lineMap x z) 1) x
      hyz : Ne ((AffineMap.lineMap x z) 1) z
      ⊢ False
    -/
    exact hyz (lineMap_apply_one _ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      t : R
      ht : Membership.mem (Set.Icc 0 1) t
      hyx : Ne ((AffineMap.lineMap x z) t) x
      hyz : Ne ((AffineMap.lineMap x z) t) z
      ho : Membership.mem (Set.Ioo 0 1) t
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) ((Affine …
    -/
  · exact ⟨t, ho, rfl⟩
    /-
      🎉 no goals
    -/


theorem Wbtw.mem_affineSpan {x y z : P} (h : Wbtw R x y z) : y ∈ line[R, x, z] := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ Membership.mem (affineSpan R (Insert.insert x (Singleton.singleton z))) y
  -/
  rcases h with ⟨r, ⟨-, rfl⟩⟩
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    r : R
    ⊢ Membership.mem (affineSpan R (Insert.insert x (Singleton.singleton z))) ((Af …
  -/
  exact lineMap_mem_affineSpan_pair _ _ _
  /-
    🎉 no goals
  -/


theorem wbtw_comm {x y z : P} : Wbtw R x y z ↔ Wbtw R z y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Wbtw R x y z) (Wbtw R z y x)
  -/
  rw [Wbtw, Wbtw, affineSegment_comm]
  /-
    🎉 no goals
  -/


alias ⟨Wbtw.symm, _⟩ := wbtw_comm


theorem sbtw_comm {x y z : P} : Sbtw R x y z ↔ Sbtw R z y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Sbtw R x y z) (Sbtw R z y x)
  -/
  rw [Sbtw, Sbtw, wbtw_comm, ← and_assoc, ← and_assoc, and_right_comm]
  /-
    🎉 no goals
  -/


alias ⟨Sbtw.symm, _⟩ := sbtw_comm


@[simp]
theorem wbtw_self_left (x y : P) : Wbtw R x x y :=
  left_mem_affineSegment _ _ _


@[simp]
theorem wbtw_self_right (x y : P) : Wbtw R x y y :=
  right_mem_affineSegment _ _ _


@[simp]
theorem wbtw_self_iff {x y : P} : Wbtw R x y x ↔ y = x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Iff (Wbtw R x y x) (Eq y x)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- Porting note: Originally `simpa [Wbtw, affineSegment] using h`
    /-
      case refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      h : Wbtw R x y x
      ⊢ Eq y x
    -/
    have ⟨_, _, h₂⟩ := h
    /-
      case refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      h : Wbtw R x y x
      w✝ : R
      left✝ : Membership.mem (Set.Icc 0 1) w✝
      h₂ : Eq ((AffineMap.lineMap x x) w✝) y
      ⊢ Eq y x
    -/
    rw [h₂.symm, lineMap_same_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      h : Eq y x
      ⊢ Wbtw R x y x
    -/
  · rw [h]
    /-
      case refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : OrderedRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      h : Eq y x
      ⊢ Wbtw R x x x
    -/
    exact wbtw_self_left R x x
    /-
      🎉 no goals
    -/


@[simp]
theorem not_sbtw_self_left (x y : P) : ¬Sbtw R x x y :=
  fun h => h.ne_left rfl


@[simp]
theorem not_sbtw_self_right (x y : P) : ¬Sbtw R x y y :=
  fun h => h.ne_right rfl


theorem Wbtw.left_ne_right_of_ne_left {x y z : P} (h : Wbtw R x y z) (hne : y ≠ x) : x ≠ z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    hne : Ne y x
    ⊢ Ne x z
  -/
  rintro rfl
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    hne : Ne y x
    h : Wbtw R x y x
    ⊢ False
  -/
  rw [wbtw_self_iff] at h
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    hne : Ne y x
    h : Eq y x
    ⊢ False
  -/
  exact hne h
  /-
    🎉 no goals
  -/


theorem Wbtw.left_ne_right_of_ne_right {x y z : P} (h : Wbtw R x y z) (hne : y ≠ z) : x ≠ z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    hne : Ne y z
    ⊢ Ne x z
  -/
  rintro rfl
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Wbtw R x y x
    hne : Ne y x
    ⊢ False
  -/
  rw [wbtw_self_iff] at h
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Eq y x
    hne : Ne y x
    ⊢ False
  -/
  exact hne h
  /-
    🎉 no goals
  -/


theorem Sbtw.left_ne_right {x y z : P} (h : Sbtw R x y z) : x ≠ z :=
  h.wbtw.left_ne_right_of_ne_left h.2.1


theorem sbtw_iff_mem_image_Ioo_and_ne [NoZeroSMulDivisors R V] {x y z : P} :
    Sbtw R x y z ↔ y ∈ lineMap x z '' Set.Ioo (0 : R) 1 ∧ x ≠ z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y z : P
    ⊢ Iff (Sbtw R x y z) (And (Membership.mem (Set.image (⇑(AffineMap.lineMap x z) …
  -/
  refine ⟨fun h => ⟨h.mem_image_Ioo, h.left_ne_right⟩, fun h => ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y z : P
    h : And (Membership.mem (Set.image (⇑(AffineMap.lineMap x z)) (Set.Ioo 0 1)) y …
    ⊢ Sbtw R x y z
  -/
  rcases h with ⟨⟨t, ht, rfl⟩, hxz⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x z : P
    hxz : Ne x z
    t : R
    ht : Membership.mem (Set.Ioo 0 1) t
    ⊢ Sbtw R x ((AffineMap.lineMap x z) t) z
  -/
  refine ⟨⟨t, Set.mem_Icc_of_Ioo ht, rfl⟩, ?_⟩
  rw [lineMap_apply, ← @vsub_ne_zero V, ← @vsub_ne_zero V _ _ _ _ z, vadd_vsub_assoc, vsub_self,
    vadd_vsub_assoc, ← neg_vsub_eq_vsub_rev z x, ← @neg_one_smul R, ← add_smul, ← sub_eq_add_neg]
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x z : P
    hxz : Ne x z
    t : R
    ht : Membership.mem (Set.Ioo 0 1) t
    ⊢ And (Ne (HAdd.hAdd (HSMul.hSMul t (VSub.vsub z x)) 0) 0) (Ne (HSMul.hSMul (H …
  -/
  simp [smul_ne_zero, sub_eq_zero, ht.1.ne.symm, ht.2.ne, hxz.symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_sbtw_self (x y : P) : ¬Sbtw R x y x :=
  fun h => h.left_ne_right rfl


theorem wbtw_swap_left_iff [NoZeroSMulDivisors R V] {x y : P} (z : P) :
    Wbtw R x y z ∧ Wbtw R y x z ↔ x = y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y z : P
    ⊢ Iff (And (Wbtw R x y z) (Wbtw R y x z)) (Eq x y)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x y z : P
      ⊢ And (Wbtw R x y z) (Wbtw R y x z) → Eq x y
    -/
  · rintro ⟨hxyz, hyxz⟩
    /-
      case mp.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x y z : P
      hxyz : Wbtw R x y z
      hyxz : Wbtw R y x z
      ⊢ Eq x y
    -/
    rcases hxyz with ⟨ty, hty, rfl⟩
    /-
      case mp.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x z : P
      ty : R
      hty : Membership.mem (Set.Icc 0 1) ty
      hyxz : Wbtw R ((AffineMap.lineMap x z) ty) x z
      ⊢ Eq x ((AffineMap.lineMap x z) ty)
    -/
    rcases hyxz with ⟨tx, htx, hx⟩
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x z : P
      ty : R
      hty : Membership.mem (Set.Icc 0 1) ty
      tx : R
      htx : Membership.mem (Set.Icc 0 1) tx
      hx : Eq ((AffineMap.lineMap ((AffineMap.lineMap x z) ty) z) tx) x
      ⊢ Eq x ((AffineMap.lineMap x z) ty)
    -/
    rw [lineMap_apply, lineMap_apply, ← add_vadd] at hx
    rw [← @vsub_eq_zero_iff_eq V, vadd_vsub, vsub_vadd_eq_vsub_sub, smul_sub, smul_smul, ← sub_smul,
      ← add_smul, smul_eq_zero] at hx
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x z : P
      ty : R
      hty : Membership.mem (Set.Icc 0 1) ty
      tx : R
      htx : Membership.mem (Set.Icc 0 1) tx
      hx : Or (Eq (HAdd.hAdd (HSub.hSub tx (HMul.hMul tx ty)) ty) 0) (Eq (VSub.vsub  …
      ⊢ Eq x ((AffineMap.lineMap x z) ty)
    -/
    rcases hx with (h | h)
      /-
        case mp.intro.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝⁴ : OrderedRing R
        inst✝³ : AddCommGroup V
        inst✝² : Module R V
        inst✝¹ : AddTorsor V P
        inst✝ : NoZeroSMulDivisors R V
        x z : P
        ty : R
        hty : Membership.mem (Set.Icc 0 1) ty
        tx : R
        htx : Membership.mem (Set.Icc 0 1) tx
        h : Eq (HAdd.hAdd (HSub.hSub tx (HMul.hMul tx ty)) ty) 0
        ⊢ Eq x ((AffineMap.lineMap x z) ty)
      -/
    · nth_rw 1 [← mul_one tx] at h
      /-
        case mp.intro.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝⁴ : OrderedRing R
        inst✝³ : AddCommGroup V
        inst✝² : Module R V
        inst✝¹ : AddTorsor V P
        inst✝ : NoZeroSMulDivisors R V
        x z : P
        ty : R
        hty : Membership.mem (Set.Icc 0 1) ty
        tx : R
        htx : Membership.mem (Set.Icc 0 1) tx
        h : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul tx 1) (HMul.hMul tx ty)) ty) 0
        ⊢ Eq x ((AffineMap.lineMap x z) ty)
      -/
      rw [← mul_sub, add_eq_zero_iff_neg_eq] at h
      have h' : ty = 0 := by
        refine le_antisymm ?_ hty.1
        rw [← h, Left.neg_nonpos_iff]
        exact mul_nonneg htx.1 (sub_nonneg.2 hty.2)
      /-
        case mp.intro.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝⁴ : OrderedRing R
        inst✝³ : AddCommGroup V
        inst✝² : Module R V
        inst✝¹ : AddTorsor V P
        inst✝ : NoZeroSMulDivisors R V
        x z : P
        ty : R
        hty : Membership.mem (Set.Icc 0 1) ty
        tx : R
        htx : Membership.mem (Set.Icc 0 1) tx
        h : Eq (Neg.neg (HMul.hMul tx (HSub.hSub 1 ty))) ty
        h' : Eq ty 0
        ⊢ Eq x ((AffineMap.lineMap x z) ty)
      -/
      simp [h']
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝⁴ : OrderedRing R
        inst✝³ : AddCommGroup V
        inst✝² : Module R V
        inst✝¹ : AddTorsor V P
        inst✝ : NoZeroSMulDivisors R V
        x z : P
        ty : R
        hty : Membership.mem (Set.Icc 0 1) ty
        tx : R
        htx : Membership.mem (Set.Icc 0 1) tx
        h : Eq (VSub.vsub z x) 0
        ⊢ Eq x ((AffineMap.lineMap x z) ty)
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case mp.intro.intro.intro.intro.intro.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝⁴ : OrderedRing R
        inst✝³ : AddCommGroup V
        inst✝² : Module R V
        inst✝¹ : AddTorsor V P
        inst✝ : NoZeroSMulDivisors R V
        x z : P
        ty : R
        hty : Membership.mem (Set.Icc 0 1) ty
        tx : R
        htx : Membership.mem (Set.Icc 0 1) tx
        h : Eq z x
        ⊢ Eq x ((AffineMap.lineMap x z) ty)
      -/
      rw [h, lineMap_same_apply]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x y z : P
      ⊢ Eq x y → And (Wbtw R x y z) (Wbtw R y x z)
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x z : P
      ⊢ And (Wbtw R x x z) (Wbtw R x x z)
    -/
    exact ⟨wbtw_self_left _ _ _, wbtw_self_left _ _ _⟩
    /-
      🎉 no goals
    -/


theorem wbtw_swap_right_iff [NoZeroSMulDivisors R V] (x : P) {y z : P} :
    Wbtw R x y z ∧ Wbtw R x z y ↔ y = z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y z : P
    ⊢ Iff (And (Wbtw R x y z) (Wbtw R x z y)) (Eq y z)
  -/
  rw [wbtw_comm, wbtw_comm (z := y), eq_comm]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y z : P
    ⊢ Iff (And (Wbtw R z y x) (Wbtw R y z x)) (Eq z y)
  -/
  exact wbtw_swap_left_iff R x
  /-
    🎉 no goals
  -/


theorem wbtw_rotate_iff [NoZeroSMulDivisors R V] (x : P) {y z : P} :
                                              /-
                                                R : Type u_1
                                                V : Type u_2
                                                P : Type u_4
                                                inst✝⁴ : OrderedRing R
                                                inst✝³ : AddCommGroup V
                                                inst✝² : Module R V
                                                inst✝¹ : AddTorsor V P
                                                inst✝ : NoZeroSMulDivisors R V
                                                x y z : P
                                                ⊢ Iff (And (Wbtw R x y z) (Wbtw R z x y)) (Eq x y)
                                              -/
    Wbtw R x y z ∧ Wbtw R z x y ↔ x = y := by rw [wbtw_comm, wbtw_swap_right_iff, eq_comm]
                                              /-
                                                🎉 no goals
                                              -/


theorem Wbtw.swap_left_iff [NoZeroSMulDivisors R V] {x y z : P} (h : Wbtw R x y z) :
                               /-
                                 R : Type u_1
                                 V : Type u_2
                                 P : Type u_4
                                 inst✝⁴ : OrderedRing R
                                 inst✝³ : AddCommGroup V
                                 inst✝² : Module R V
                                 inst✝¹ : AddTorsor V P
                                 inst✝ : NoZeroSMulDivisors R V
                                 x y z : P
                                 h : Wbtw R x y z
                                 ⊢ Iff (Wbtw R y x z) (Eq x y)
                               -/
    Wbtw R y x z ↔ x = y := by rw [← wbtw_swap_left_iff R z, and_iff_right h]
                               /-
                                 🎉 no goals
                               -/


theorem Wbtw.swap_right_iff [NoZeroSMulDivisors R V] {x y z : P} (h : Wbtw R x y z) :
                               /-
                                 R : Type u_1
                                 V : Type u_2
                                 P : Type u_4
                                 inst✝⁴ : OrderedRing R
                                 inst✝³ : AddCommGroup V
                                 inst✝² : Module R V
                                 inst✝¹ : AddTorsor V P
                                 inst✝ : NoZeroSMulDivisors R V
                                 x y z : P
                                 h : Wbtw R x y z
                                 ⊢ Iff (Wbtw R x z y) (Eq y z)
                               -/
    Wbtw R x z y ↔ y = z := by rw [← wbtw_swap_right_iff R x, and_iff_right h]
                               /-
                                 🎉 no goals
                               -/


theorem Wbtw.rotate_iff [NoZeroSMulDivisors R V] {x y z : P} (h : Wbtw R x y z) :
                               /-
                                 R : Type u_1
                                 V : Type u_2
                                 P : Type u_4
                                 inst✝⁴ : OrderedRing R
                                 inst✝³ : AddCommGroup V
                                 inst✝² : Module R V
                                 inst✝¹ : AddTorsor V P
                                 inst✝ : NoZeroSMulDivisors R V
                                 x y z : P
                                 h : Wbtw R x y z
                                 ⊢ Iff (Wbtw R z x y) (Eq x y)
                               -/
    Wbtw R z x y ↔ x = y := by rw [← wbtw_rotate_iff R x, and_iff_right h]
                               /-
                                 🎉 no goals
                               -/


theorem Sbtw.not_swap_left [NoZeroSMulDivisors R V] {x y z : P} (h : Sbtw R x y z) :
    ¬Wbtw R y x z := fun hs => h.left_ne (h.wbtw.swap_left_iff.1 hs)


theorem Sbtw.not_swap_right [NoZeroSMulDivisors R V] {x y z : P} (h : Sbtw R x y z) :
    ¬Wbtw R x z y := fun hs => h.ne_right (h.wbtw.swap_right_iff.1 hs)


theorem Sbtw.not_rotate [NoZeroSMulDivisors R V] {x y z : P} (h : Sbtw R x y z) : ¬Wbtw R z x y :=
  fun hs => h.left_ne (h.wbtw.rotate_iff.1 hs)


@[simp]
theorem wbtw_lineMap_iff [NoZeroSMulDivisors R V] {x y : P} {r : R} :
    Wbtw R x (lineMap x y r) y ↔ x = y ∨ r ∈ Set.Icc (0 : R) 1 := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y : P
    r : R
    ⊢ Iff (Wbtw R x ((AffineMap.lineMap x y) r) y) (Or (Eq x y) (Membership.mem (S …
  -/
  by_cases hxy : x = y
    /-
      case pos
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x y : P
      r : R
      hxy : Eq x y
      ⊢ Iff (Wbtw R x ((AffineMap.lineMap x y) r) y) (Or (Eq x y) (Membership.mem (S …
    -/
  · rw [hxy, lineMap_same_apply]
    /-
      case pos
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : OrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      x y : P
      r : R
      hxy : Eq x y
      ⊢ Iff (Wbtw R y y y) (Or (Eq y y) (Membership.mem (Set.Icc 0 1) r))
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y : P
    r : R
    hxy : Not (Eq x y)
    ⊢ Iff (Wbtw R x ((AffineMap.lineMap x y) r) y) (Or (Eq x y) (Membership.mem (S …
  -/
  rw [or_iff_right hxy, Wbtw, affineSegment, (lineMap_injective R hxy).mem_set_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem sbtw_lineMap_iff [NoZeroSMulDivisors R V] {x y : P} {r : R} :
    Sbtw R x (lineMap x y r) y ↔ x ≠ y ∧ r ∈ Set.Ioo (0 : R) 1 := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y : P
    r : R
    ⊢ Iff (Sbtw R x ((AffineMap.lineMap x y) r) y) (And (Ne x y) (Membership.mem ( …
  -/
  rw [sbtw_iff_mem_image_Ioo_and_ne, and_comm, and_congr_right]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y : P
    r : R
    ⊢ Ne x y → Iff (Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioo  …
  -/
  intro hxy
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    x y : P
    r : R
    hxy : Ne x y
    ⊢ Iff (Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioo 0 1)) ((A …
  -/
  rw [(lineMap_injective R hxy).mem_set_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem wbtw_mul_sub_add_iff [NoZeroDivisors R] {x y r : R} :
    Wbtw R x (r * (y - x) + x) y ↔ x = y ∨ r ∈ Set.Icc (0 : R) 1 :=
  wbtw_lineMap_iff


@[simp]
theorem sbtw_mul_sub_add_iff [NoZeroDivisors R] {x y r : R} :
    Sbtw R x (r * (y - x) + x) y ↔ x ≠ y ∧ r ∈ Set.Ioo (0 : R) 1 :=
  sbtw_lineMap_iff


@[simp]
theorem wbtw_zero_one_iff {x : R} : Wbtw R 0 x 1 ↔ x ∈ Set.Icc (0 : R) 1 := by
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Wbtw R 0 x 1) (Membership.mem (Set.Icc 0 1) x)
  -/
  rw [Wbtw, affineSegment, Set.mem_image]
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (Set.Icc 0 1) x_1) (Eq ((AffineMa …
  -/
  simp_rw [lineMap_apply_ring]
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (Set.Icc 0 1) x_1) (Eq (HAdd.hAdd …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem wbtw_one_zero_iff {x : R} : Wbtw R 1 x 0 ↔ x ∈ Set.Icc (0 : R) 1 := by
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Wbtw R 1 x 0) (Membership.mem (Set.Icc 0 1) x)
  -/
  rw [wbtw_comm, wbtw_zero_one_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem sbtw_zero_one_iff {x : R} : Sbtw R 0 x 1 ↔ x ∈ Set.Ioo (0 : R) 1 := by
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Sbtw R 0 x 1) (Membership.mem (Set.Ioo 0 1) x)
  -/
  rw [Sbtw, wbtw_zero_one_iff, Set.mem_Icc, Set.mem_Ioo]
  exact
    ⟨fun h => ⟨h.1.1.lt_of_ne (Ne.symm h.2.1), h.1.2.lt_of_ne h.2.2⟩, fun h =>
      ⟨⟨h.1.le, h.2.le⟩, h.1.ne', h.2.ne⟩⟩


@[simp]
theorem sbtw_one_zero_iff {x : R} : Sbtw R 1 x 0 ↔ x ∈ Set.Ioo (0 : R) 1 := by
  /-
    R : Type u_1
    inst✝ : OrderedRing R
    x : R
    ⊢ Iff (Sbtw R 1 x 0) (Membership.mem (Set.Ioo 0 1) x)
  -/
  rw [sbtw_comm, sbtw_zero_one_iff]
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_left {w x y z : P} (h₁ : Wbtw R w y z) (h₂ : Wbtw R w x y) : Wbtw R w x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R w y z
    h₂ : Wbtw R w x y
    ⊢ Wbtw R w x z
  -/
  rcases h₁ with ⟨t₁, ht₁, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    h₂ : Wbtw R w x ((AffineMap.lineMap w z) t₁)
    ⊢ Wbtw R w x z
  -/
  rcases h₂ with ⟨t₂, ht₂, rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    t₂ : R
    ht₂ : Membership.mem (Set.Icc 0 1) t₂
    ⊢ Wbtw R w ((AffineMap.lineMap w ((AffineMap.lineMap w z) t₁)) t₂) z
  -/
  refine ⟨t₂ * t₁, ⟨mul_nonneg ht₂.1 ht₁.1, mul_le_one₀ ht₂.2 ht₁.1 ht₁.2⟩, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    t₂ : R
    ht₂ : Membership.mem (Set.Icc 0 1) t₂
    ⊢ Eq ((AffineMap.lineMap w z) (HMul.hMul t₂ t₁)) ((AffineMap.lineMap w ((Affin …
  -/
  rw [lineMap_apply, lineMap_apply, lineMap_vsub_left, smul_smul]
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_right {w x y z : P} (h₁ : Wbtw R w x z) (h₂ : Wbtw R x y z) : Wbtw R w y z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R w x z
    h₂ : Wbtw R x y z
    ⊢ Wbtw R w y z
  -/
  rw [wbtw_comm] at *
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : OrderedRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R z x w
    h₂ : Wbtw R z y x
    ⊢ Wbtw R z y w
  -/
  exact h₁.trans_left h₂
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_sbtw_left [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Wbtw R w y z)
    (h₂ : Sbtw R w x y) : Sbtw R w x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R w y z
    h₂ : Sbtw R w x y
    ⊢ Sbtw R w x z
  -/
  refine ⟨h₁.trans_left h₂.wbtw, h₂.ne_left, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R w y z
    h₂ : Sbtw R w x y
    ⊢ Ne x z
  -/
  rintro rfl
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y : P
    h₂ : Sbtw R w x y
    h₁ : Wbtw R w y x
    ⊢ False
  -/
  exact h₂.right_ne ((wbtw_swap_right_iff R w).1 ⟨h₁, h₂.wbtw⟩)
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_sbtw_right [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Wbtw R w x z)
    (h₂ : Sbtw R x y z) : Sbtw R w y z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R w x z
    h₂ : Sbtw R x y z
    ⊢ Sbtw R w y z
  -/
  rw [wbtw_comm] at *
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R z x w
    h₂ : Sbtw R x y z
    ⊢ Sbtw R w y z
  -/
  rw [sbtw_comm] at *
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R z x w
    h₂ : Sbtw R z y x
    ⊢ Sbtw R z y w
  -/
  exact h₁.trans_sbtw_left h₂
  /-
    🎉 no goals
  -/


theorem Sbtw.trans_left [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Sbtw R w y z)
    (h₂ : Sbtw R w x y) : Sbtw R w x z :=
  h₁.wbtw.trans_sbtw_left h₂


theorem Sbtw.trans_right [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Sbtw R w x z)
    (h₂ : Sbtw R x y z) : Sbtw R w y z :=
  h₁.wbtw.trans_sbtw_right h₂


theorem Wbtw.trans_left_ne [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Wbtw R w y z)
    (h₂ : Wbtw R w x y) (h : y ≠ z) : x ≠ z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R w y z
    h₂ : Wbtw R w x y
    h : Ne y z
    ⊢ Ne x z
  -/
  rintro rfl
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y : P
    h₂ : Wbtw R w x y
    h₁ : Wbtw R w y x
    h : Ne y x
    ⊢ False
  -/
  exact h (h₁.swap_right_iff.1 h₂)
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_right_ne [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Wbtw R w x z)
    (h₂ : Wbtw R x y z) (h : w ≠ x) : w ≠ y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x y z : P
    h₁ : Wbtw R w x z
    h₂ : Wbtw R x y z
    h : Ne w x
    ⊢ Ne w y
  -/
  rintro rfl
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : OrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    w x z : P
    h₁ : Wbtw R w x z
    h : Ne w x
    h₂ : Wbtw R x w z
    ⊢ False
  -/
  exact h (h₁.swap_left_iff.1 h₂)
  /-
    🎉 no goals
  -/


theorem Sbtw.trans_wbtw_left_ne [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Sbtw R w y z)
    (h₂ : Wbtw R w x y) : x ≠ z :=
  h₁.wbtw.trans_left_ne h₂ h₁.ne_right


theorem Sbtw.trans_wbtw_right_ne [NoZeroSMulDivisors R V] {w x y z : P} (h₁ : Sbtw R w x z)
    (h₂ : Wbtw R x y z) : w ≠ y :=
  h₁.wbtw.trans_right_ne h₂ h₁.left_ne


theorem Sbtw.affineCombination_of_mem_affineSpan_pair [NoZeroDivisors R] [NoZeroSMulDivisors R V]
    {ι : Type*} {p : ι → P} (ha : AffineIndependent R p) {w w₁ w₂ : ι → R} {s : Finset ι}
    (hw : ∑ i ∈ s, w i = 1) (hw₁ : ∑ i ∈ s, w₁ i = 1) (hw₂ : ∑ i ∈ s, w₂ i = 1)
    (h : s.affineCombination R p w ∈
      line[R, s.affineCombination R p w₁, s.affineCombination R p w₂])
    {i : ι} (his : i ∈ s) (hs : Sbtw R (w₁ i) (w i) (w₂ i)) :
    Sbtw R (s.affineCombination R p w₁) (s.affineCombination R p w)
      (s.affineCombination R p w₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    h : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R s …
    i : ι
    his : Membership.mem s i
    hs : Sbtw R (w₁ i) (w i) (w₂ i)
    ⊢ Sbtw R ((Finset.affineCombination R s p) w₁) ((Finset.affineCombination R s  …
  -/
  rw [affineCombination_mem_affineSpan_pair ha hw hw₁ hw₂] at h
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    h : Exists fun r => ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HMul. …
    i : ι
    his : Membership.mem s i
    hs : Sbtw R (w₁ i) (w i) (w₂ i)
    ⊢ Sbtw R ((Finset.affineCombination R s p) w₁) ((Finset.affineCombination R s  …
  -/
  rcases h with ⟨r, hr⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    hs : Sbtw R (w₁ i) (w i) (w₂ i)
    r : R
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HMul.hMul r (HSub.hS …
    ⊢ Sbtw R ((Finset.affineCombination R s p) w₁) ((Finset.affineCombination R s  …
  -/
  rw [hr i his, sbtw_mul_sub_add_iff] at hs
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HMul.hMul r (HSub.hS …
    ⊢ Sbtw R ((Finset.affineCombination R s p) w₁) ((Finset.affineCombination R s  …
  -/
  change ∀ i ∈ s, w i = (r • (w₂ - w₁) + w₁) i at hr
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HSMul.hSMul r (HSub. …
    ⊢ Sbtw R ((Finset.affineCombination R s p) w₁) ((Finset.affineCombination R s  …
  -/
  rw [s.affineCombination_congr hr fun _ _ => rfl]
  rw [← s.weightedVSub_vadd_affineCombination, s.weightedVSub_const_smul,
    ← s.affineCombination_vsub, ← lineMap_apply, sbtw_lineMap_iff, and_iff_left hs.2,
    ← @vsub_ne_zero V, s.affineCombination_vsub]
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HSMul.hSMul r (HSub. …
    ⊢ Ne ((s.weightedVSub p) (HSub.hSub w₁ w₂)) 0
  -/
  intro hz
  have hw₁w₂ : (∑ i ∈ s, (w₁ - w₂) i) = 0 := by
    simp_rw [Pi.sub_apply, Finset.sum_sub_distrib, hw₁, hw₂, sub_self]
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HSMul.hSMul r (HSub. …
    hz : Eq ((s.weightedVSub p) (HSub.hSub w₁ w₂)) 0
    hw₁w₂ : Eq (s.sum fun i => HSub.hSub w₁ w₂ i) 0
    ⊢ False
  -/
  refine hs.1 ?_
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HSMul.hSMul r (HSub. …
    hz : Eq ((s.weightedVSub p) (HSub.hSub w₁ w₂)) 0
    hw₁w₂ : Eq (s.sum fun i => HSub.hSub w₁ w₂ i) 0
    ⊢ Eq (w₁ i) (w₂ i)
  -/
  have ha' := ha s (w₁ - w₂) hw₁w₂ hz i his
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁵ : OrderedRing R
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module R V
    inst✝² : AddTorsor V P
    inst✝¹ : NoZeroDivisors R
    inst✝ : NoZeroSMulDivisors R V
    ι : Type u_6
    p : ι → P
    ha : AffineIndependent R p
    w w₁ w₂ : ι → R
    s : Finset ι
    hw : Eq (s.sum fun i => w i) 1
    hw₁ : Eq (s.sum fun i => w₁ i) 1
    hw₂ : Eq (s.sum fun i => w₂ i) 1
    i : ι
    his : Membership.mem s i
    r : R
    hs : And (Ne (w₁ i) (w₂ i)) (Membership.mem (Set.Ioo 0 1) r)
    hr : ∀ (i : ι), Membership.mem s i → Eq (w i) (HAdd.hAdd (HSMul.hSMul r (HSub. …
    hz : Eq ((s.weightedVSub p) (HSub.hSub w₁ w₂)) 0
    hw₁w₂ : Eq (s.sum fun i => HSub.hSub w₁ w₂ i) 0
    ha' : Eq (HSub.hSub w₁ w₂ i) 0
    ⊢ Eq (w₁ i) (w₂ i)
  -/
  rwa [Pi.sub_apply, sub_eq_zero] at ha'
  /-
    🎉 no goals
  -/


theorem Wbtw.sameRay_vsub {x y z : P} (h : Wbtw R x y z) : SameRay R (y -ᵥ x) (z -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ SameRay R (VSub.vsub y x) (VSub.vsub z y)
  -/
  rcases h with ⟨t, ⟨ht0, ht1⟩, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ⊢ SameRay R (VSub.vsub ((AffineMap.lineMap x z) t) x) (VSub.vsub z ((AffineMap …
  -/
  simp_rw [lineMap_apply]
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub z x)) x) x) (VSu …
  -/
  rcases ht0.lt_or_eq with (ht0' | rfl); swap; · simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case intro.intro.intro.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ht0' : LT.lt 0 t
    ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub z x)) x) x) (VSu …
  -/
  rcases ht1.lt_or_eq with (ht1' | rfl); swap; · simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ht0' : LT.lt 0 t
    ht1' : LT.lt t 1
    ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub z x)) x) x) (VSu …
  -/
  refine Or.inr (Or.inr ⟨1 - t, t, sub_pos.2 ht1', ht0', ?_⟩)
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ht0' : LT.lt 0 t
    ht1' : LT.lt t 1
    ⊢ Eq (HSMul.hSMul (HSub.hSub 1 t) (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub …
  -/
  simp only [vadd_vsub, smul_smul, vsub_vadd_eq_vsub_sub, smul_sub, ← sub_smul]
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    ht0' : LT.lt 0 t
    ht1' : LT.lt t 1
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HSub.hSub 1 t) t) (VSub.vsub z x)) (HSMul.hSMul  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem Wbtw.sameRay_vsub_left {x y z : P} (h : Wbtw R x y z) : SameRay R (y -ᵥ x) (z -ᵥ x) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ SameRay R (VSub.vsub y x) (VSub.vsub z x)
  -/
  rcases h with ⟨t, ⟨ht0, _⟩, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    t : R
    ht0 : LE.le 0 t
    right✝ : LE.le t 1
    ⊢ SameRay R (VSub.vsub ((AffineMap.lineMap x z) t) x) (VSub.vsub z x)
  -/
  simpa [lineMap_apply] using SameRay.sameRay_nonneg_smul_left (z -ᵥ x) ht0
  /-
    🎉 no goals
  -/


theorem Wbtw.sameRay_vsub_right {x y z : P} (h : Wbtw R x y z) : SameRay R (z -ᵥ x) (z -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ SameRay R (VSub.vsub z x) (VSub.vsub z y)
  -/
  rcases h with ⟨t, ⟨_, ht1⟩, rfl⟩
  simpa [lineMap_apply, vsub_vadd_eq_vsub_sub, sub_smul] using
    SameRay.sameRay_nonneg_smul_right (z -ᵥ x) (sub_nonneg.2 ht1)


/-- Suppose lines from two vertices of a triangle to interior points of the opposite side meet at
`p`. Then `p` lies in the interior of the first (and by symmetry the other) segment from a
vertex to the point on the opposite side. -/
theorem sbtw_of_sbtw_of_sbtw_of_mem_affineSpan_pair [NoZeroSMulDivisors R V]
    {t : Affine.Triangle R P} {i₁ i₂ i₃ : Fin 3} (h₁₂ : i₁ ≠ i₂) {p₁ p₂ p : P}
    (h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)) (h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃))
    (h₁' : p ∈ line[R, t.points i₁, p₁]) (h₂' : p ∈ line[R, t.points i₂, p₂]) :
    Sbtw R (t.points i₁) p p₁ := by
  -- Should not be needed; see comments on local instances in `Data.Sign`.
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  letI : DecidableRel ((· < ·) : R → R → Prop) := LinearOrderedRing.decidableLT
  have h₁₃ : i₁ ≠ i₃ := by
    rintro rfl
    simp at h₂
  have h₂₃ : i₂ ≠ i₃ := by
    rintro rfl
    simp at h₁
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  have h3 : ∀ i : Fin 3, i = i₁ ∨ i = i₂ ∨ i = i₃ := by omega
  have hu : (Finset.univ : Finset (Fin 3)) = {i₁, i₂, i₃} := by
    clear h₁ h₂ h₁' h₂'
    -- Porting note: Originally `decide!`
    revert i₁ i₂ i₃; decide
  have hp : p ∈ affineSpan R (Set.range t.points) := by
    have hle : line[R, t.points i₁, p₁] ≤ affineSpan R (Set.range t.points) := by
      refine affineSpan_pair_le_of_mem_of_mem (mem_affineSpan R (Set.mem_range_self _)) ?_
      have hle : line[R, t.points i₂, t.points i₃] ≤ affineSpan R (Set.range t.points) := by
        refine affineSpan_mono R ?_
        simp [Set.insert_subset_iff]
      rw [AffineSubspace.le_def'] at hle
      exact hle _ h₁.wbtw.mem_affineSpan
    rw [AffineSubspace.le_def'] at hle
    exact hle _ h₁'
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  have h₁i := h₁.mem_image_Ioo
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    h₁i : Membership.mem (Set.image (⇑(AffineMap.lineMap (t.points i₂) (t.points i …
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  have h₂i := h₂.mem_image_Ioo
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    h₁i : Membership.mem (Set.image (⇑(AffineMap.lineMap (t.points i₂) (t.points i …
    h₂i : Membership.mem (Set.image (⇑(AffineMap.lineMap (t.points i₁) (t.points i …
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  rw [Set.mem_image] at h₁i h₂i
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₁ p₂ p : P
    h₁ : Sbtw R (t.points i₂) p₁ (t.points i₃)
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    h₁i : Exists fun x => And (Membership.mem (Set.Ioo 0 1) x) (Eq ((AffineMap.lin …
    h₂i : Exists fun x => And (Membership.mem (Set.Ioo 0 1) x) (Eq ((AffineMap.lin …
    ⊢ Sbtw R (t.points i₁) p p₁
  -/
  rcases h₁i with ⟨r₁, ⟨hr₁0, hr₁1⟩, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p₂ p : P
    h₂ : Sbtw R (t.points i₁) p₂ (t.points i₃)
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    h₂i : Exists fun x => And (Membership.mem (Set.Ioo 0 1) x) (Eq ((AffineMap.lin …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    ⊢ Sbtw R (t.points i₁) p ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁)
  -/
  rcases h₂i with ⟨r₂, ⟨hr₂0, hr₂1⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    p : P
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    hp : Membership.mem (affineSpan R (Set.range t.points)) p
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    h₁' : Membership.mem (affineSpan R (Insert.insert (t.points i₁) (Singleton.sin …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    ⊢ Sbtw R (t.points i₁) p ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁)
  -/
  rcases eq_affineCombination_of_mem_affineSpan_of_fintype hp with ⟨w, hw, rfl⟩
  have h₁s :=
    sign_eq_of_affineCombination_mem_affineSpan_single_lineMap t.independent hw (Finset.mem_univ _)
      (Finset.mem_univ _) (Finset.mem_univ _) h₁₂ h₁₃ h₂₃ hr₁0 hr₁1 h₁'
  have h₂s :=
    sign_eq_of_affineCombination_mem_affineSpan_single_lineMap t.independent hw (Finset.mem_univ _)
      (Finset.mem_univ _) (Finset.mem_univ _) h₁₂.symm h₂₃ h₁₃ hr₂0 hr₂1 h₂'
  rw [← Finset.univ.affineCombination_affineCombinationSingleWeights R t.points
      (Finset.mem_univ i₁),
    ← Finset.univ.affineCombination_affineCombinationLineMapWeights t.points (Finset.mem_univ _)
      (Finset.mem_univ _)] at h₁' ⊢
  refine
    Sbtw.affineCombination_of_mem_affineSpan_pair t.independent hw
      (Finset.univ.sum_affineCombinationSingleWeights R (Finset.mem_univ _))
      (Finset.univ.sum_affineCombinationLineMapWeights (Finset.mem_univ _) (Finset.mem_univ _) _)
      h₁' (Finset.mem_univ i₁) ?_
  rw [Finset.affineCombinationSingleWeights_apply_self,
    Finset.affineCombinationLineMapWeights_apply_of_ne h₁₂ h₁₃, sbtw_one_zero_iff]
  have hs : ∀ i : Fin 3, SignType.sign (w i) = SignType.sign (w i₃) := by
    intro i
    rcases h3 i with (rfl | rfl | rfl)
    · exact h₂s
    · exact h₁s
    · rfl
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), Eq (SignType.sign (w i)) (SignType.sign (w i₃))
    ⊢ Membership.mem (Set.Ioo 0 1) (w i₁)
  -/
  have hss : SignType.sign (∑ i, w i) = 1 := by simp [hw]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), Eq (SignType.sign (w i)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (Finset.univ.sum fun i => w i)) 1
    ⊢ Membership.mem (Set.Ioo 0 1) (w i₁)
  -/
  have hs' := sign_sum Finset.univ_nonempty (SignType.sign (w i₃)) fun i _ => hs i
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), Eq (SignType.sign (w i)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (Finset.univ.sum fun i => w i)) 1
    hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
    ⊢ Membership.mem (Set.Ioo 0 1) (w i₁)
  -/
  rw [hs'] at hss
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), Eq (SignType.sign (w i)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (w i₃)) 1
    hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
    ⊢ Membership.mem (Set.Ioo 0 1) (w i₁)
  -/
  simp_rw [hss, sign_eq_one_iff] at hs
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (w i₃)) 1
    hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), LT.lt 0 (w i)
    ⊢ Membership.mem (Set.Ioo 0 1) (w i₁)
  -/
  refine ⟨hs i₁, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq (Finset.univ.sum fun i => w i) 1
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (w i₃)) 1
    hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), LT.lt 0 (w i)
    ⊢ LT.lt (w i₁) 1
  -/
  rw [hu] at hw
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝⁴ : LinearOrderedRing R
    inst✝³ : AddCommGroup V
    inst✝² : Module R V
    inst✝¹ : AddTorsor V P
    inst✝ : NoZeroSMulDivisors R V
    t : Affine.Triangle R P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
    hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
    r₁ : R
    hr₁0 : LT.lt 0 r₁
    hr₁1 : LT.lt r₁ 1
    h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
    r₂ : R
    hr₂0 : LT.lt 0 r₂
    hr₂1 : LT.lt r₂ 1
    h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
    w : Fin (HAdd.hAdd 2 1) → R
    hw : Eq ((Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i₃))).sum fu …
    hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
    h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
    h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
    h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
    h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
    hss : Eq (SignType.sign (w i₃)) 1
    hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
    hs : ∀ (i : Fin 3), LT.lt 0 (w i)
    ⊢ LT.lt (w i₁) 1
  -/
  rw [Finset.sum_insert, Finset.sum_insert, Finset.sum_singleton] at hw
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : LinearOrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      t : Affine.Triangle R P
      i₁ i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
      r₁ : R
      hr₁0 : LT.lt 0 r₁
      hr₁1 : LT.lt r₁ 1
      h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
      r₂ : R
      hr₂0 : LT.lt 0 r₂
      hr₂1 : LT.lt r₂ 1
      h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
      w : Fin (HAdd.hAdd 2 1) → R
      hw : Eq (HAdd.hAdd (w i₁) (HAdd.hAdd (w i₂) (w i₃))) 1
      hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
      h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
      h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
      h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
      h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
      hss : Eq (SignType.sign (w i₃)) 1
      hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
      hs : ∀ (i : Fin 3), LT.lt 0 (w i)
      ⊢ LT.lt (w i₁) 1
    -/
  · by_contra hle
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : LinearOrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      t : Affine.Triangle R P
      i₁ i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
      r₁ : R
      hr₁0 : LT.lt 0 r₁
      hr₁1 : LT.lt r₁ 1
      h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
      r₂ : R
      hr₂0 : LT.lt 0 r₂
      hr₂1 : LT.lt r₂ 1
      h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
      w : Fin (HAdd.hAdd 2 1) → R
      hw : Eq (HAdd.hAdd (w i₁) (HAdd.hAdd (w i₂) (w i₃))) 1
      hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
      h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
      h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
      h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
      h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
      hss : Eq (SignType.sign (w i₃)) 1
      hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
      hs : ∀ (i : Fin 3), LT.lt 0 (w i)
      hle : Not (LT.lt (w i₁) 1)
      ⊢ False
    -/
    rw [not_lt] at hle
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : LinearOrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      t : Affine.Triangle R P
      i₁ i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
      r₁ : R
      hr₁0 : LT.lt 0 r₁
      hr₁1 : LT.lt r₁ 1
      h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
      r₂ : R
      hr₂0 : LT.lt 0 r₂
      hr₂1 : LT.lt r₂ 1
      h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
      w : Fin (HAdd.hAdd 2 1) → R
      hw : Eq (HAdd.hAdd (w i₁) (HAdd.hAdd (w i₂) (w i₃))) 1
      hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
      h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
      h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
      h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
      h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
      hss : Eq (SignType.sign (w i₃)) 1
      hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
      hs : ∀ (i : Fin 3), LT.lt 0 (w i)
      hle : LE.le 1 (w i₁)
      ⊢ False
    -/
    exact (hle.trans_lt (lt_add_of_pos_right _ (Left.add_pos (hs i₂) (hs i₃)))).ne' hw
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : LinearOrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      t : Affine.Triangle R P
      i₁ i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
      r₁ : R
      hr₁0 : LT.lt 0 r₁
      hr₁1 : LT.lt r₁ 1
      h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
      r₂ : R
      hr₂0 : LT.lt 0 r₂
      hr₂1 : LT.lt r₂ 1
      h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
      w : Fin (HAdd.hAdd 2 1) → R
      hw : Eq (HAdd.hAdd (w i₁) ((Insert.insert i₂ (Singleton.singleton i₃)).sum fun …
      hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
      h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
      h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
      h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
      h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
      hss : Eq (SignType.sign (w i₃)) 1
      hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
      hs : ∀ (i : Fin 3), LT.lt 0 (w i)
      ⊢ Not (Membership.mem (Singleton.singleton i₃) i₂)
    -/
  · simpa using h₂₃
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝⁴ : LinearOrderedRing R
      inst✝³ : AddCommGroup V
      inst✝² : Module R V
      inst✝¹ : AddTorsor V P
      inst✝ : NoZeroSMulDivisors R V
      t : Affine.Triangle R P
      i₁ i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      this : DecidableRel fun x1 x2 => LT.lt x1 x2 := LinearOrderedRing.decidableLT
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h3 : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      hu : Eq Finset.univ (Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i …
      r₁ : R
      hr₁0 : LT.lt 0 r₁
      hr₁1 : LT.lt r₁ 1
      h₁ : Sbtw R (t.points i₂) ((AffineMap.lineMap (t.points i₂) (t.points i₃)) r₁) …
      r₂ : R
      hr₂0 : LT.lt 0 r₂
      hr₂1 : LT.lt r₂ 1
      h₂ : Sbtw R (t.points i₁) ((AffineMap.lineMap (t.points i₁) (t.points i₃)) r₂) …
      w : Fin (HAdd.hAdd 2 1) → R
      hw : Eq ((Insert.insert i₁ (Insert.insert i₂ (Singleton.singleton i₃))).sum fu …
      hp : Membership.mem (affineSpan R (Set.range t.points)) ((Finset.affineCombina …
      h₁' : Membership.mem (affineSpan R (Insert.insert ((Finset.affineCombination R …
      h₂' : Membership.mem (affineSpan R (Insert.insert (t.points i₂) (Singleton.sin …
      h₁s : Eq (SignType.sign (w i₂)) (SignType.sign (w i₃))
      h₂s : Eq (SignType.sign (w i₁)) (SignType.sign (w i₃))
      hss : Eq (SignType.sign (w i₃)) 1
      hs' : Eq (SignType.sign (Finset.univ.sum fun i => w i)) (SignType.sign (w i₃))
      hs : ∀ (i : Fin 3), LT.lt 0 (w i)
      ⊢ Not (Membership.mem (Insert.insert i₂ (Singleton.singleton i₃)) i₁)
    -/
  · simpa [not_or] using ⟨h₁₂, h₁₃⟩
    /-
      🎉 no goals
    -/


theorem wbtw_iff_left_eq_or_right_mem_image_Ici {x y z : P} :
    Wbtw R x y z ↔ x = y ∨ z ∈ lineMap x y '' Set.Ici (1 : R) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Wbtw R x y z) (Or (Eq x y) (Membership.mem (Set.image (⇑(AffineMap.line …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Wbtw R x y z
      ⊢ Or (Eq x y) (Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ici 1 …
    -/
  · rcases h with ⟨r, ⟨hr0, hr1⟩, rfl⟩
    /-
      case refine_1.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      r : R
      hr0 : LE.le 0 r
      hr1 : LE.le r 1
      ⊢ Or (Eq x ((AffineMap.lineMap x z) r)) (Membership.mem (Set.image (⇑(AffineMa …
    -/
    rcases hr0.lt_or_eq with (hr0' | rfl)
      /-
        case refine_1.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        r : R
        hr0 : LE.le 0 r
        hr1 : LE.le r 1
        hr0' : LT.lt 0 r
        ⊢ Or (Eq x ((AffineMap.lineMap x z) r)) (Membership.mem (Set.image (⇑(AffineMa …
      -/
    · rw [Set.mem_image]
      /-
        case refine_1.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        r : R
        hr0 : LE.le 0 r
        hr1 : LE.le r 1
        hr0' : LT.lt 0 r
        ⊢ Or (Eq x ((AffineMap.lineMap x z) r)) (Exists fun x_1 => And (Membership.mem …
      -/
      refine .inr ⟨r⁻¹, (one_le_inv₀ hr0').2 hr1, ?_⟩
      /-
        case refine_1.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        r : R
        hr0 : LE.le 0 r
        hr1 : LE.le r 1
        hr0' : LT.lt 0 r
        ⊢ Eq ((AffineMap.lineMap x ((AffineMap.lineMap x z) r)) (Inv.inv r)) z
      -/
      simp only [lineMap_apply, smul_smul, vadd_vsub]
      /-
        case refine_1.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        r : R
        hr0 : LE.le 0 r
        hr1 : LE.le r 1
        hr0' : LT.lt 0 r
        ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul (HMul.hMul (Inv.inv r) r) (VSub.vsub z x)) x) z
      -/
      rw [inv_mul_cancel₀ hr0'.ne', one_smul, vsub_vadd]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        hr0 : LE.le 0 0
        hr1 : LE.le 0 1
        ⊢ Or (Eq x ((AffineMap.lineMap x z) 0)) (Membership.mem (Set.image (⇑(AffineMa …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Or (Eq x y) (Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ici …
      ⊢ Wbtw R x y z
    -/
  · rcases h with (rfl | ⟨r, ⟨hr, rfl⟩⟩)
      /-
        case refine_2.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x z : P
        ⊢ Wbtw R x x z
      -/
    · exact wbtw_self_left _ _ _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        r : R
        hr : Membership.mem (Set.Ici 1) r
        ⊢ Wbtw R x y ((AffineMap.lineMap x y) r)
      -/
    · rw [Set.mem_Ici] at hr
      /-
        case refine_2.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        r : R
        hr : LE.le 1 r
        ⊢ Wbtw R x y ((AffineMap.lineMap x y) r)
      -/
      refine ⟨r⁻¹, ⟨inv_nonneg.2 (zero_le_one.trans hr), inv_le_one_of_one_le₀ hr⟩, ?_⟩
      /-
        case refine_2.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        r : R
        hr : LE.le 1 r
        ⊢ Eq ((AffineMap.lineMap x ((AffineMap.lineMap x y) r)) (Inv.inv r)) y
      -/
      simp only [lineMap_apply, smul_smul, vadd_vsub]
      /-
        case refine_2.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        r : R
        hr : LE.le 1 r
        ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul (HMul.hMul (Inv.inv r) r) (VSub.vsub y x)) x) y
      -/
      rw [inv_mul_cancel₀ (one_pos.trans_le hr).ne', one_smul, vsub_vadd]
      /-
        🎉 no goals
      -/


theorem Wbtw.right_mem_image_Ici_of_left_ne {x y z : P} (h : Wbtw R x y z) (hne : x ≠ y) :
    z ∈ lineMap x y '' Set.Ici (1 : R) :=
  (wbtw_iff_left_eq_or_right_mem_image_Ici.1 h).resolve_left hne


theorem Wbtw.right_mem_affineSpan_of_left_ne {x y z : P} (h : Wbtw R x y z) (hne : x ≠ y) :
    z ∈ line[R, x, y] := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    hne : Ne x y
    ⊢ Membership.mem (affineSpan R (Insert.insert x (Singleton.singleton y))) z
  -/
  rcases h.right_mem_image_Ici_of_left_ne hne with ⟨r, ⟨-, rfl⟩⟩
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    hne : Ne x y
    r : R
    h : Wbtw R x y ((AffineMap.lineMap x y) r)
    ⊢ Membership.mem (affineSpan R (Insert.insert x (Singleton.singleton y))) ((Af …
  -/
  exact lineMap_mem_affineSpan_pair _ _ _
  /-
    🎉 no goals
  -/


theorem sbtw_iff_left_ne_and_right_mem_image_Ioi {x y z : P} :
    Sbtw R x y z ↔ x ≠ y ∧ z ∈ lineMap x y '' Set.Ioi (1 : R) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Sbtw R x y z) (And (Ne x y) (Membership.mem (Set.image (⇑(AffineMap.lin …
  -/
  refine ⟨fun h => ⟨h.left_ne, ?_⟩, fun h => ?_⟩
    /-
      case refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Sbtw R x y z
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioi 1)) z
    -/
  · obtain ⟨r, ⟨hr, rfl⟩⟩ := h.wbtw.right_mem_image_Ici_of_left_ne h.left_ne
    /-
      case refine_1.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      r : R
      hr : Membership.mem (Set.Ici 1) r
      h : Sbtw R x y ((AffineMap.lineMap x y) r)
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioi 1)) ((AffineMa …
    -/
    rw [Set.mem_Ici] at hr
    /-
      case refine_1.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      r : R
      hr : LE.le 1 r
      h : Sbtw R x y ((AffineMap.lineMap x y) r)
      ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioi 1)) ((AffineMa …
    -/
    rcases hr.lt_or_eq with (hrlt | rfl)
      /-
        case refine_1.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        r : R
        hr : LE.le 1 r
        h : Sbtw R x y ((AffineMap.lineMap x y) r)
        hrlt : LT.lt 1 r
        ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioi 1)) ((AffineMa …
      -/
    · exact Set.mem_image_of_mem _ hrlt
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        hr : LE.le 1 1
        h : Sbtw R x y ((AffineMap.lineMap x y) 1)
        ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Ioi 1)) ((AffineMa …
      -/
    · exfalso
      /-
        case refine_1.intro.intro.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x y : P
        hr : LE.le 1 1
        h : Sbtw R x y ((AffineMap.lineMap x y) 1)
        ⊢ False
      -/
      simp at h
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : And (Ne x y) (Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (Set.Io …
      ⊢ Sbtw R x y z
    -/
  · rcases h with ⟨hne, r, hr, rfl⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      hne : Ne x y
      r : R
      hr : Membership.mem (Set.Ioi 1) r
      ⊢ Sbtw R x y ((AffineMap.lineMap x y) r)
    -/
    rw [Set.mem_Ioi] at hr
    refine
      ⟨wbtw_iff_left_eq_or_right_mem_image_Ici.2
          (Or.inr (Set.mem_image_of_mem _ (Set.mem_of_mem_of_subset hr Set.Ioi_subset_Ici_self))),
        hne.symm, ?_⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      hne : Ne x y
      r : R
      hr : LT.lt 1 r
      ⊢ Ne y ((AffineMap.lineMap x y) r)
    -/
    rw [lineMap_apply, ← @vsub_ne_zero V, vsub_vadd_eq_vsub_sub]
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      hne : Ne x y
      r : R
      hr : LT.lt 1 r
      ⊢ Ne (HSub.hSub (VSub.vsub y x) (HSMul.hSMul r (VSub.vsub y x))) 0
    -/
    nth_rw 1 [← one_smul R (y -ᵥ x)]
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      hne : Ne x y
      r : R
      hr : LT.lt 1 r
      ⊢ Ne (HSub.hSub (HSMul.hSMul 1 (VSub.vsub y x)) (HSMul.hSMul r (VSub.vsub y x) …
    -/
    rw [← sub_smul, smul_ne_zero_iff, vsub_ne_zero, sub_ne_zero]
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y : P
      hne : Ne x y
      r : R
      hr : LT.lt 1 r
      ⊢ And (Ne 1 r) (Ne y x)
    -/
    exact ⟨hr.ne, hne.symm⟩
    /-
      🎉 no goals
    -/


theorem Sbtw.right_mem_image_Ioi {x y z : P} (h : Sbtw R x y z) :
    z ∈ lineMap x y '' Set.Ioi (1 : R) :=
  (sbtw_iff_left_ne_and_right_mem_image_Ioi.1 h).2


theorem Sbtw.right_mem_affineSpan {x y z : P} (h : Sbtw R x y z) : z ∈ line[R, x, y] :=
  h.wbtw.right_mem_affineSpan_of_left_ne h.left_ne


theorem wbtw_iff_right_eq_or_left_mem_image_Ici {x y z : P} :
    Wbtw R x y z ↔ z = y ∨ x ∈ lineMap z y '' Set.Ici (1 : R) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Wbtw R x y z) (Or (Eq z y) (Membership.mem (Set.image (⇑(AffineMap.line …
  -/
  rw [wbtw_comm, wbtw_iff_left_eq_or_right_mem_image_Ici]
  /-
    🎉 no goals
  -/


theorem Wbtw.left_mem_image_Ici_of_right_ne {x y z : P} (h : Wbtw R x y z) (hne : z ≠ y) :
    x ∈ lineMap z y '' Set.Ici (1 : R) :=
  h.symm.right_mem_image_Ici_of_left_ne hne


theorem Wbtw.left_mem_affineSpan_of_right_ne {x y z : P} (h : Wbtw R x y z) (hne : z ≠ y) :
    x ∈ line[R, z, y] :=
  h.symm.right_mem_affineSpan_of_left_ne hne


theorem sbtw_iff_right_ne_and_left_mem_image_Ioi {x y z : P} :
    Sbtw R x y z ↔ z ≠ y ∧ x ∈ lineMap z y '' Set.Ioi (1 : R) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Sbtw R x y z) (And (Ne z y) (Membership.mem (Set.image (⇑(AffineMap.lin …
  -/
  rw [sbtw_comm, sbtw_iff_left_ne_and_right_mem_image_Ioi]
  /-
    🎉 no goals
  -/


theorem Sbtw.left_mem_image_Ioi {x y z : P} (h : Sbtw R x y z) :
    x ∈ lineMap z y '' Set.Ioi (1 : R) :=
  h.symm.right_mem_image_Ioi


theorem Sbtw.left_mem_affineSpan {x y z : P} (h : Sbtw R x y z) : x ∈ line[R, z, y] :=
  h.symm.right_mem_affineSpan


lemma AffineSubspace.right_mem_of_wbtw {s : AffineSubspace R P} (hxyz : Wbtw R x y z) (hx : x ∈ s)
    (hy : y ∈ s) (hxy : x ≠ y) : z ∈ s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    s : AffineSubspace R P
    hxyz : Wbtw R x y z
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ Membership.mem s z
  -/
  obtain ⟨ε, -, rfl⟩ := hxyz
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    s : AffineSubspace R P
    hx : Membership.mem s x
    ε : R
    hy : Membership.mem s ((AffineMap.lineMap x z) ε)
    hxy : Ne x ((AffineMap.lineMap x z) ε)
    ⊢ Membership.mem s z
  -/
  have hε : ε ≠ 0 := by rintro rfl; simp at hxy
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    s : AffineSubspace R P
    hx : Membership.mem s x
    ε : R
    hy : Membership.mem s ((AffineMap.lineMap x z) ε)
    hxy : Ne x ((AffineMap.lineMap x z) ε)
    hε : Ne ε 0
    ⊢ Membership.mem s z
  -/
  simpa [hε] using lineMap_mem ε⁻¹ hx hy
  /-
    🎉 no goals
  -/


theorem wbtw_smul_vadd_smul_vadd_of_nonneg_of_le (x : P) (v : V) {r₁ r₂ : R} (hr₁ : 0 ≤ r₁)
    (hr₂ : r₁ ≤ r₂) : Wbtw R x (r₁ • v +ᵥ x) (r₂ • v +ᵥ x) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le r₁ r₂
    ⊢ Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ v) x)
  -/
  refine ⟨r₁ / r₂, ⟨div_nonneg hr₁ (hr₁.trans hr₂), div_le_one_of_le₀ hr₂ (hr₁.trans hr₂)⟩, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le r₁ r₂
    ⊢ Eq ((AffineMap.lineMap x (HVAdd.hVAdd (HSMul.hSMul r₂ v) x)) (HDiv.hDiv r₁ r …
  -/
  by_cases h : r₁ = 0; · simp [h]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le r₁ r₂
    h : Not (Eq r₁ 0)
    ⊢ Eq ((AffineMap.lineMap x (HVAdd.hVAdd (HSMul.hSMul r₂ v) x)) (HDiv.hDiv r₁ r …
  -/
  simp [lineMap_apply, smul_smul, ((hr₁.lt_of_ne' h).trans_le hr₂).ne.symm]
  /-
    🎉 no goals
  -/


theorem wbtw_or_wbtw_smul_vadd_of_nonneg (x : P) (v : V) {r₁ r₂ : R} (hr₁ : 0 ≤ r₁) (hr₂ : 0 ≤ r₂) :
    Wbtw R x (r₁ • v +ᵥ x) (r₂ • v +ᵥ x) ∨ Wbtw R x (r₂ • v +ᵥ x) (r₁ • v +ᵥ x) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le 0 r₂
    ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
  -/
  rcases le_total r₁ r₂ with (h | h)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le 0 r₁
      hr₂ : LE.le 0 r₂
      h : LE.le r₁ r₂
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
    -/
  · exact Or.inl (wbtw_smul_vadd_smul_vadd_of_nonneg_of_le x v hr₁ h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le 0 r₁
      hr₂ : LE.le 0 r₂
      h : LE.le r₂ r₁
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
    -/
  · exact Or.inr (wbtw_smul_vadd_smul_vadd_of_nonneg_of_le x v hr₂ h)
    /-
      🎉 no goals
    -/


theorem wbtw_smul_vadd_smul_vadd_of_nonpos_of_le (x : P) (v : V) {r₁ r₂ : R} (hr₁ : r₁ ≤ 0)
    (hr₂ : r₂ ≤ r₁) : Wbtw R x (r₁ • v +ᵥ x) (r₂ • v +ᵥ x) := by
  convert wbtw_smul_vadd_smul_vadd_of_nonneg_of_le x (-v) (Left.nonneg_neg_iff.2 hr₁)
      (neg_le_neg_iff.2 hr₂) using 1 <;>
    /-
      case h.e'_9
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le r₁ 0
      hr₂ : LE.le r₂ r₁
      ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul (Neg.neg r₁) …
    -/
    /-
      🎉 no goals
    -/
    rw [neg_smul_neg]
    /-
      🎉 no goals
    -/


theorem wbtw_or_wbtw_smul_vadd_of_nonpos (x : P) (v : V) {r₁ r₂ : R} (hr₁ : r₁ ≤ 0) (hr₂ : r₂ ≤ 0) :
    Wbtw R x (r₁ • v +ᵥ x) (r₂ • v +ᵥ x) ∨ Wbtw R x (r₂ • v +ᵥ x) (r₁ • v +ᵥ x) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le r₁ 0
    hr₂ : LE.le r₂ 0
    ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
  -/
  rcases le_total r₁ r₂ with (h | h)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le r₁ 0
      hr₂ : LE.le r₂ 0
      h : LE.le r₁ r₂
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
    -/
  · exact Or.inr (wbtw_smul_vadd_smul_vadd_of_nonpos_of_le x v hr₂ h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le r₁ 0
      hr₂ : LE.le r₂ 0
      h : LE.le r₂ r₁
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) (HVAdd.hVAdd (HSMul.hSMul r₂ …
    -/
  · exact Or.inl (wbtw_smul_vadd_smul_vadd_of_nonpos_of_le x v hr₁ h)
    /-
      🎉 no goals
    -/


theorem wbtw_smul_vadd_smul_vadd_of_nonpos_of_nonneg (x : P) (v : V) {r₁ r₂ : R} (hr₁ : r₁ ≤ 0)
    (hr₂ : 0 ≤ r₂) : Wbtw R (r₁ • v +ᵥ x) x (r₂ • v +ᵥ x) := by
  convert wbtw_smul_vadd_smul_vadd_of_nonneg_of_le (r₁ • v +ᵥ x) v (Left.nonneg_neg_iff.2 hr₁)
      (neg_le_sub_iff_le_add.2 ((le_add_iff_nonneg_left r₁).2 hr₂)) using 1 <;>
    /-
      case h.e'_9
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      r₁ r₂ : R
      hr₁ : LE.le r₁ 0
      hr₂ : LE.le 0 r₂
      ⊢ Eq x (HVAdd.hVAdd (HSMul.hSMul (Neg.neg r₁) v) (HVAdd.hVAdd (HSMul.hSMul r₁  …
    -/
    /-
      🎉 no goals
    -/
    simp [sub_smul, ← add_vadd]
    /-
      🎉 no goals
    -/


theorem wbtw_smul_vadd_smul_vadd_of_nonneg_of_nonpos (x : P) (v : V) {r₁ r₂ : R} (hr₁ : 0 ≤ r₁)
    (hr₂ : r₂ ≤ 0) : Wbtw R (r₁ • v +ᵥ x) x (r₂ • v +ᵥ x) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le r₂ 0
    ⊢ Wbtw R (HVAdd.hVAdd (HSMul.hSMul r₁ v) x) x (HVAdd.hVAdd (HSMul.hSMul r₂ v) x)
  -/
  rw [wbtw_comm]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    r₁ r₂ : R
    hr₁ : LE.le 0 r₁
    hr₂ : LE.le r₂ 0
    ⊢ Wbtw R (HVAdd.hVAdd (HSMul.hSMul r₂ v) x) x (HVAdd.hVAdd (HSMul.hSMul r₁ v) x)
  -/
  exact wbtw_smul_vadd_smul_vadd_of_nonpos_of_nonneg x v hr₂ hr₁
  /-
    🎉 no goals
  -/


theorem Wbtw.trans_left_right {w x y z : P} (h₁ : Wbtw R w y z) (h₂ : Wbtw R w x y) :
    Wbtw R x y z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R w y z
    h₂ : Wbtw R w x y
    ⊢ Wbtw R x y z
  -/
  rcases h₁ with ⟨t₁, ht₁, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    h₂ : Wbtw R w x ((AffineMap.lineMap w z) t₁)
    ⊢ Wbtw R x ((AffineMap.lineMap w z) t₁) z
  -/
  rcases h₂ with ⟨t₂, ht₂, rfl⟩
  refine
    ⟨(t₁ - t₂ * t₁) / (1 - t₂ * t₁),
      ⟨div_nonneg (sub_nonneg.2 (mul_le_of_le_one_left ht₁.1 ht₂.2))
          (sub_nonneg.2 (mul_le_one₀ ht₂.2 ht₁.1 ht₁.2)), div_le_one_of_le₀
            (sub_le_sub_right ht₁.2 _) (sub_nonneg.2 (mul_le_one₀ ht₂.2 ht₁.1 ht₁.2))⟩,
      ?_⟩
  simp only [lineMap_apply, smul_smul, ← add_vadd, vsub_vadd_eq_vsub_sub, smul_sub, ← sub_smul,
    ← add_smul, vadd_vsub, vadd_right_cancel_iff, div_mul_eq_mul_div, div_sub_div_same]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    t₂ : R
    ht₂ : Membership.mem (Set.Icc 0 1) t₂
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HSub.hSub t₁ (HMul.hMul t₂ …
  -/
  nth_rw 1 [← mul_one (t₁ - t₂ * t₁)]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    t₂ : R
    ht₂ : Membership.mem (Set.Icc 0 1) t₂
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul (HSub.hSub t₁ (H …
  -/
  rw [← mul_sub, mul_div_assoc]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w z : P
    t₁ : R
    ht₁ : Membership.mem (Set.Icc 0 1) t₁
    t₂ : R
    ht₂ : Membership.mem (Set.Icc 0 1) t₂
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ (HMul.hMul t₂ t₁)) (HDiv …
  -/
  by_cases h : 1 - t₂ * t₁ = 0
    /-
      case pos
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      w z : P
      t₁ : R
      ht₁ : Membership.mem (Set.Icc 0 1) t₁
      t₂ : R
      ht₂ : Membership.mem (Set.Icc 0 1) t₂
      h : Eq (HSub.hSub 1 (HMul.hMul t₂ t₁)) 0
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ (HMul.hMul t₂ t₁)) (HDiv …
    -/
  · rw [sub_eq_zero, eq_comm] at h
    /-
      case pos
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      w z : P
      t₁ : R
      ht₁ : Membership.mem (Set.Icc 0 1) t₁
      t₂ : R
      ht₂ : Membership.mem (Set.Icc 0 1) t₂
      h : Eq (HMul.hMul t₂ t₁) 1
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ (HMul.hMul t₂ t₁)) (HDiv …
    -/
    rw [h]
    /-
      case pos
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      w z : P
      t₁ : R
      ht₁ : Membership.mem (Set.Icc 0 1) t₁
      t₂ : R
      ht₂ : Membership.mem (Set.Icc 0 1) t₂
      h : Eq (HMul.hMul t₂ t₁) 1
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ 1) (HDiv.hDiv (HSub.hSub …
    -/
    suffices t₁ = 1 by simp [this]
    exact
      eq_of_le_of_not_lt ht₁.2 fun ht₁lt =>
        (mul_lt_one_of_nonneg_of_lt_one_right ht₂.2 ht₁.1 ht₁lt).ne h
    /-
      case neg
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      w z : P
      t₁ : R
      ht₁ : Membership.mem (Set.Icc 0 1) t₁
      t₂ : R
      ht₂ : Membership.mem (Set.Icc 0 1) t₂
      h : Not (Eq (HSub.hSub 1 (HMul.hMul t₂ t₁)) 0)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ (HMul.hMul t₂ t₁)) (HDiv …
    -/
  · rw [div_self h]
    /-
      case neg
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      w z : P
      t₁ : R
      ht₁ : Membership.mem (Set.Icc 0 1) t₁
      t₂ : R
      ht₂ : Membership.mem (Set.Icc 0 1) t₂
      h : Not (Eq (HSub.hSub 1 (HMul.hMul t₂ t₁)) 0)
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HSub.hSub t₁ (HMul.hMul t₂ t₁)) 1) (H …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


theorem Wbtw.trans_right_left {w x y z : P} (h₁ : Wbtw R w x z) (h₂ : Wbtw R x y z) :
    Wbtw R w x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R w x z
    h₂ : Wbtw R x y z
    ⊢ Wbtw R w x y
  -/
  rw [wbtw_comm] at *
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    w x y z : P
    h₁ : Wbtw R z x w
    h₂ : Wbtw R z y x
    ⊢ Wbtw R y x w
  -/
  exact h₁.trans_left_right h₂
  /-
    🎉 no goals
  -/


theorem Sbtw.trans_left_right {w x y z : P} (h₁ : Sbtw R w y z) (h₂ : Sbtw R w x y) :
    Sbtw R x y z :=
  ⟨h₁.wbtw.trans_left_right h₂.wbtw, h₂.right_ne, h₁.ne_right⟩


theorem Sbtw.trans_right_left {w x y z : P} (h₁ : Sbtw R w x z) (h₂ : Sbtw R x y z) :
    Sbtw R w x y :=
  ⟨h₁.wbtw.trans_right_left h₂.wbtw, h₁.ne_left, h₂.left_ne⟩


theorem Wbtw.collinear {x y z : P} (h : Wbtw R x y z) : Collinear R ({x, y, z} : Set P) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ Collinear R (Insert.insert x (Insert.insert y (Singleton.singleton z)))
  -/
  rw [collinear_iff_exists_forall_eq_smul_vadd]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ Exists fun p₀ => Exists fun v => ∀ (p : P), Membership.mem (Insert.insert x  …
  -/
  refine ⟨x, z -ᵥ x, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    ⊢ ∀ (p : P), Membership.mem (Insert.insert x (Insert.insert y (Singleton.singl …
  -/
  intro p hp
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    p : P
    hp : Membership.mem (Insert.insert x (Insert.insert y (Singleton.singleton z)) …
    ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub z x)) x)
  -/
  simp_rw [Set.mem_insert_iff, Set.mem_singleton_iff] at hp
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Wbtw R x y z
    p : P
    hp : Or (Eq p x) (Or (Eq p y) (Eq p z))
    ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub z x)) x)
  -/
  rcases hp with (rfl | rfl | rfl)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      y z p : P
      h : Wbtw R p y z
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub z p)) p)
    -/
  · refine ⟨0, ?_⟩
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      y z p : P
      h : Wbtw R p y z
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul 0 (VSub.vsub z p)) p)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z p : P
      h : Wbtw R x p z
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub z x)) x)
    -/
  · rcases h with ⟨t, -, rfl⟩
    /-
      case inr.inl.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x z : P
      t : R
      ⊢ Exists fun r => Eq ((AffineMap.lineMap x z) t) (HVAdd.hVAdd (HSMul.hSMul r ( …
    -/
    exact ⟨t, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y p : P
      h : Wbtw R x y p
      ⊢ Exists fun r => Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p x)) x)
    -/
  · refine ⟨1, ?_⟩
    /-
      case inr.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y p : P
      h : Wbtw R x y p
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul 1 (VSub.vsub p x)) x)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem Collinear.wbtw_or_wbtw_or_wbtw {x y z : P} (h : Collinear R ({x, y, z} : Set P)) :
    Wbtw R x y z ∨ Wbtw R y z x ∨ Wbtw R z x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Collinear R (Insert.insert x (Insert.insert y (Singleton.singleton z)))
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  rw [collinear_iff_of_mem (Set.mem_insert _ _)] at h
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : Exists fun v => ∀ (p : P), Membership.mem (Insert.insert x (Insert.insert  …
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  rcases h with ⟨v, h⟩
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    v : V
    h : ∀ (p : P), Membership.mem (Insert.insert x (Insert.insert y (Singleton.sin …
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  simp_rw [Set.mem_insert_iff, Set.mem_singleton_iff] at h
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    v : V
    h : ∀ (p : P), Or (Eq p x) (Or (Eq p y) (Eq p z)) → Exists fun r => Eq p (HVAd …
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  have hy := h y (Or.inr (Or.inl rfl))
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    v : V
    h : ∀ (p : P), Or (Eq p x) (Or (Eq p y) (Eq p z)) → Exists fun r => Eq p (HVAd …
    hy : Exists fun r => Eq y (HVAdd.hVAdd (HSMul.hSMul r v) x)
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  have hz := h z (Or.inr (Or.inr rfl))
  /-
    case intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    v : V
    h : ∀ (p : P), Or (Eq p x) (Or (Eq p y) (Eq p z)) → Exists fun r => Eq p (HVAd …
    hy : Exists fun r => Eq y (HVAdd.hVAdd (HSMul.hSMul r v) x)
    hz : Exists fun r => Eq z (HVAdd.hVAdd (HSMul.hSMul r v) x)
    ⊢ Or (Wbtw R x y z) (Or (Wbtw R y z x) (Wbtw R z x y))
  -/
  rcases hy with ⟨ty, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x z : P
    v : V
    hz : Exists fun r => Eq z (HVAdd.hVAdd (HSMul.hSMul r v) x)
    ty : R
    h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
    ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) z) (Or (Wbtw R (HVAdd.hVAdd  …
  -/
  rcases hz with ⟨tz, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x : P
    v : V
    ty tz : R
    h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
    ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
  -/
  rcases lt_trichotomy ty 0 with (hy0 | rfl | hy0)
    /-
      case intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      ty tz : R
      h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
      hy0 : LT.lt ty 0
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
    -/
  · rcases lt_trichotomy tz 0 with (hz0 | rfl | hz0)
      /-
        case intro.intro.intro.inl.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt ty 0
        hz0 : LT.lt tz 0
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
    · rw [wbtw_comm (z := x)]
      /-
        case intro.intro.intro.inl.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt ty 0
        hz0 : LT.lt tz 0
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
      rw [← or_assoc]
      /-
        case intro.intro.intro.inl.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt ty 0
        hz0 : LT.lt tz 0
        ⊢ Or (Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMu …
      -/
      exact Or.inl (wbtw_or_wbtw_smul_vadd_of_nonpos _ _ hy0.le hz0.le)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inl.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty : R
        hy0 : LT.lt ty 0
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul 0  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inl.inr.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt ty 0
        hz0 : LT.lt 0 tz
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
    · exact Or.inr (Or.inr (wbtw_smul_vadd_smul_vadd_of_nonneg_of_nonpos _ _ hz0.le hy0.le))
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      tz : R
      h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul 0 v) x)) (Eq p  …
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul 0 v) x) (HVAdd.hVAdd (HSMul.hSMul tz  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.inr.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x : P
      v : V
      ty tz : R
      h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
      hy0 : LT.lt 0 ty
      ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
    -/
  · rcases lt_trichotomy tz 0 with (hz0 | rfl | hz0)
      /-
        case intro.intro.intro.inr.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt 0 ty
        hz0 : LT.lt tz 0
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
    · refine Or.inr (Or.inr (wbtw_smul_vadd_smul_vadd_of_nonpos_of_nonneg _ _ hz0.le hy0.le))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inr.inr.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty : R
        hy0 : LT.lt 0 ty
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul 0  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.inr.inr.inr.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt 0 ty
        hz0 : LT.lt 0 tz
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
    · rw [wbtw_comm (z := x)]
      /-
        case intro.intro.intro.inr.inr.inr.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt 0 ty
        hz0 : LT.lt 0 tz
        ⊢ Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMul tz …
      -/
      rw [← or_assoc]
      /-
        case intro.intro.intro.inr.inr.inr.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        x : P
        v : V
        ty tz : R
        h : ∀ (p : P), Or (Eq p x) (Or (Eq p (HVAdd.hVAdd (HSMul.hSMul ty v) x)) (Eq p …
        hy0 : LT.lt 0 ty
        hz0 : LT.lt 0 tz
        ⊢ Or (Or (Wbtw R x (HVAdd.hVAdd (HSMul.hSMul ty v) x) (HVAdd.hVAdd (HSMul.hSMu …
      -/
      exact Or.inl (wbtw_or_wbtw_smul_vadd_of_nonneg _ _ hy0.le hz0.le)
      /-
        🎉 no goals
      -/


theorem wbtw_iff_sameRay_vsub {x y z : P} : Wbtw R x y z ↔ SameRay R (y -ᵥ x) (z -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    ⊢ Iff (Wbtw R x y z) (SameRay R (VSub.vsub y x) (VSub.vsub z y))
  -/
  refine ⟨Wbtw.sameRay_vsub, fun h => ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y z : P
    h : SameRay R (VSub.vsub y x) (VSub.vsub z y)
    ⊢ Wbtw R x y z
  -/
  rcases h with (h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Eq (VSub.vsub y x) 0
      ⊢ Wbtw R x y z
    -/
  · rw [vsub_eq_zero_iff_eq] at h
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Eq y x
      ⊢ Wbtw R x y z
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Eq (VSub.vsub z y) 0
      ⊢ Wbtw R x y z
    -/
  · rw [vsub_eq_zero_iff_eq] at h
    /-
      case inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      h : Eq z y
      ⊢ Wbtw R x y z
    -/
    simp [h]
    /-
      🎉 no goals
    -/
  · refine
      ⟨r₂ / (r₁ + r₂),
        ⟨div_nonneg hr₂.le (add_nonneg hr₁.le hr₂.le),
          div_le_one_of_le₀ (le_add_of_nonneg_left hr₁.le) (add_nonneg hr₁.le hr₂.le)⟩,
        ?_⟩
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub y x)) (HSMul.hSMul r₂ (VSub.vsub z y))
      ⊢ Eq ((AffineMap.lineMap x z) (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂))) y
    -/
    have h' : z = r₂⁻¹ • r₁ • (y -ᵥ x) +ᵥ y := by simp [h, hr₂.ne']
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub y x)) (HSMul.hSMul r₂ (VSub.vsub z y))
      h' : Eq z (HVAdd.hVAdd (HSMul.hSMul (Inv.inv r₂) (HSMul.hSMul r₁ (VSub.vsub y  …
      ⊢ Eq ((AffineMap.lineMap x z) (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂))) y
    -/
    rw [eq_comm]
    simp only [lineMap_apply, h', vadd_vsub_assoc, smul_smul, ← add_smul, eq_vadd_iff_vsub_eq,
      smul_add]
    /-
      case inr.inr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub y x)) (HSMul.hSMul r₂ (VSub.vsub z y))
      h' : Eq z (HVAdd.hVAdd (HSMul.hSMul (Inv.inv r₂) (HSMul.hSMul r₁ (VSub.vsub y  …
      ⊢ Eq (VSub.vsub y x) (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HDiv.hDiv r₂ (HAdd.hA …
    -/
    convert (one_smul R (y -ᵥ x)).symm
    /-
      case h.e'_3.h.e'_5
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub y x)) (HSMul.hSMul r₂ (VSub.vsub z y))
      h' : Eq z (HVAdd.hVAdd (HSMul.hSMul (Inv.inv r₂) (HSMul.hSMul r₁ (VSub.vsub y  …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂)) (HMul.hMul (Inv.in …
    -/
    field_simp [(add_pos hr₁ hr₂).ne', hr₂.ne']
    /-
      case h.e'_3.h.e'_5
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      x y z : P
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub y x)) (HSMul.hSMul r₂ (VSub.vsub z y))
      h' : Eq z (HVAdd.hVAdd (HSMul.hSMul (Inv.inv r₂) (HSMul.hSMul r₁ (VSub.vsub y  …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul r₂ r₁) (HAdd.hAdd r₁ r₂)) (HMul.hMul r₂  …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem wbtw_pointReflection (x y : P) : Wbtw R y x (pointReflection R x y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Wbtw R y x ((AffineEquiv.pointReflection R x) y)
  -/
  refine ⟨2⁻¹, ⟨by norm_num, by norm_num⟩, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq ((AffineMap.lineMap y ((AffineEquiv.pointReflection R x) y)) (Inv.inv 2)) x
  -/
  rw [lineMap_apply, pointReflection_apply, vadd_vsub_assoc, ← two_smul R (x -ᵥ y)]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul (Inv.inv 2) (HSMul.hSMul 2 (VSub.vsub x y))) y) x
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sbtw_pointReflection_of_ne {x y : P} (h : x ≠ y) : Sbtw R y x (pointReflection R x y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Ne x y
    ⊢ Sbtw R y x ((AffineEquiv.pointReflection R x) y)
  -/
  refine ⟨wbtw_pointReflection _ _ _, h, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Ne x y
    ⊢ Ne x ((AffineEquiv.pointReflection R x) y)
  -/
  nth_rw 1 [← pointReflection_self R x]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Ne x y
    ⊢ Ne ((AffineEquiv.pointReflection R x) x) ((AffineEquiv.pointReflection R x) y)
  -/
  exact (pointReflection_involutive R x).injective.ne h
  /-
    🎉 no goals
  -/


theorem wbtw_midpoint (x y : P) : Wbtw R x (midpoint R x y) y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Wbtw R x (midpoint R x y) y
  -/
  convert wbtw_pointReflection R (midpoint R x y) x
  /-
    case h.e'_10
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    ⊢ Eq y ((AffineEquiv.pointReflection R (midpoint R x y)) x)
  -/
  rw [pointReflection_midpoint_left]
  /-
    🎉 no goals
  -/


theorem sbtw_midpoint_of_ne {x y : P} (h : x ≠ y) : Sbtw R x (midpoint R x y) y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h : Ne x y
    ⊢ Sbtw R x (midpoint R x y) y
  -/
  have h : midpoint R x y ≠ x := by simp [h]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h✝ : Ne x y
    h : Ne (midpoint R x y) x
    ⊢ Sbtw R x (midpoint R x y) y
  -/
  convert sbtw_pointReflection_of_ne R h
  /-
    case h.e'_10
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    x y : P
    h✝ : Ne x y
    h : Ne (midpoint R x y) x
    ⊢ Eq y ((AffineEquiv.pointReflection R (midpoint R x y)) x)
  -/
  rw [pointReflection_midpoint_left]
  /-
    🎉 no goals
  -/


