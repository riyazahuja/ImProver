/-- Characterizing uniformities associated to a (generalized) distance function `D`
in terms of the elements of the uniformity. -/
theorem uniformity_dist_of_mem_uniformity [LinearOrder β] {U : Filter (α × α)} (z : β)
    (D : α → α → β) (H : ∀ s, s ∈ U ↔ ∃ ε > z, ∀ {a b : α}, D a b < ε → (a, b) ∈ s) :
    U = ⨅ ε > z, 𝓟 { p : α × α | D p.1 p.2 < ε } :=
                                 /-
                                   α : Type u
                                   β : Type v
                                   inst✝ : LinearOrder β
                                   U : Filter (Prod α α)
                                   z : β
                                   D : α → α → β
                                   H : ∀ (s : Set (Prod α α)), Iff (Membership.mem U s) (Exists fun ε => And (GT. …
                                   s : Set (Prod α α)
                                   ⊢ Iff (Membership.mem U s) (Exists fun i => And (GT.gt i z) (HasSubset.Subset  …
                                 -/
  HasBasis.eq_biInf ⟨fun s => by simp only [H, subset_def, Prod.forall, mem_setOf]⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- `EDist α` means that `α` is equipped with an extended distance. -/
@[ext]
class EDist (α : Type*) where
  edist : α → α → ℝ≥0∞


/-- Creating a uniform space from an extended distance. -/
def uniformSpaceOfEDist (edist : α → α → ℝ≥0∞) (edist_self : ∀ x : α, edist x x = 0)
    (edist_comm : ∀ x y : α, edist x y = edist y x)
    (edist_triangle : ∀ x y z : α, edist x z ≤ edist x y + edist y z) : UniformSpace α :=
  .ofFun edist edist_self edist_comm edist_triangle fun ε ε0 =>
    ⟨ε / 2, ENNReal.half_pos ε0.ne', fun _ h₁ _ h₂ =>
      (ENNReal.add_lt_add h₁ h₂).trans_eq (ENNReal.add_halves _)⟩

-- the uniform structure is embedded in the emetric space structure
-- to avoid instance diamond issues. See Note [forgetful inheritance].

/-- Extended (pseudo) metric spaces, with an extended distance `edist` possibly taking the
value ∞

Each pseudo_emetric space induces a canonical `UniformSpace` and hence a canonical
`TopologicalSpace`.
This is enforced in the type class definition, by extending the `UniformSpace` structure. When
instantiating a `PseudoEMetricSpace` structure, the uniformity fields are not necessary, they
will be filled in by default. There is a default value for the uniformity, that can be substituted
in cases of interest, for instance when instantiating a `PseudoEMetricSpace` structure
on a product.

Continuity of `edist` is proved in `Topology.Instances.ENNReal`
-/
class PseudoEMetricSpace (α : Type u) extends EDist α : Type u where
  edist_self : ∀ x : α, edist x x = 0
  edist_comm : ∀ x y : α, edist x y = edist y x
  edist_triangle : ∀ x y z : α, edist x z ≤ edist x y + edist y z
  toUniformSpace : UniformSpace α := uniformSpaceOfEDist edist edist_self edist_comm edist_triangle
  uniformity_edist : 𝓤 α = ⨅ ε > 0, 𝓟 { p : α × α | edist p.1 p.2 < ε } := by rfl


/-- Two pseudo emetric space structures with the same edistance function coincide. -/
@[ext]
protected theorem PseudoEMetricSpace.ext {α : Type*} {m m' : PseudoEMetricSpace α}
    (h : m.toEDist = m'.toEDist) : m = m' := by
  /-
    α : Type u_2
    m m' : PseudoEMetricSpace α
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq m m'
  -/
  cases' m with ed  _ _ _ U hU
  /-
    case mk
    α : Type u_2
    m' : PseudoEMetricSpace α
    ed : EDist α
    edist_self✝ : ∀ (x : α), Eq (EDist.edist x x) 0
    edist_comm✝ : ∀ (x y : α), Eq (EDist.edist x y) (EDist.edist y x)
    edist_triangle✝ : ∀ (x y z : α), LE.le (EDist.edist x z) (HAdd.hAdd (EDist.edi …
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq (PseudoEMetricSpace.mk edist_self✝ edist_comm✝ edist_triangle✝ U hU) m'
  -/
  cases' m' with ed' _ _ _ U' hU'
  /-
    case mk.mk
    α : Type u_2
    ed : EDist α
    edist_self✝¹ : ∀ (x : α), Eq (EDist.edist x x) 0
    edist_comm✝¹ : ∀ (x y : α), Eq (EDist.edist x y) (EDist.edist y x)
    edist_triangle✝¹ : ∀ (x y z : α), LE.le (EDist.edist x z) (HAdd.hAdd (EDist.ed …
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    ed' : EDist α
    edist_self✝ : ∀ (x : α), Eq (EDist.edist x x) 0
    edist_comm✝ : ∀ (x y : α), Eq (EDist.edist x y) (EDist.edist y x)
    edist_triangle✝ : ∀ (x y z : α), LE.le (EDist.edist x z) (HAdd.hAdd (EDist.edi …
    U' : UniformSpace α
    hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq (PseudoEMetricSpace.mk edist_self✝¹ edist_comm✝¹ edist_triangle✝¹ U hU) ( …
  -/
  congr 1
  /-
    case mk.mk.e_toUniformSpace
    α : Type u_2
    ed : EDist α
    edist_self✝¹ : ∀ (x : α), Eq (EDist.edist x x) 0
    edist_comm✝¹ : ∀ (x y : α), Eq (EDist.edist x y) (EDist.edist y x)
    edist_triangle✝¹ : ∀ (x y z : α), LE.le (EDist.edist x z) (HAdd.hAdd (EDist.ed …
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    ed' : EDist α
    edist_self✝ : ∀ (x : α), Eq (EDist.edist x x) 0
    edist_comm✝ : ∀ (x y : α), Eq (EDist.edist x y) (EDist.edist y x)
    edist_triangle✝ : ∀ (x y z : α), LE.le (EDist.edist x z) (HAdd.hAdd (EDist.edi …
    U' : UniformSpace α
    hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq U U'
  -/
  exact UniformSpace.ext (((show ed = ed' from h) ▸ hU).trans hU'.symm)
  /-
    🎉 no goals
  -/


/-- Triangle inequality for the extended distance -/
theorem edist_triangle_left (x y z : α) : edist x y ≤ edist z x + edist z y := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y z : α
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (EDist.edist z x) (EDist.edist z y))
  -/
  rw [edist_comm z]; apply edist_triangle
                     /-
                       🎉 no goals
                     -/


theorem edist_triangle_right (x y z : α) : edist x y ≤ edist x z + edist y z := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y z : α
    ⊢ LE.le (EDist.edist x y) (HAdd.hAdd (EDist.edist x z) (EDist.edist y z))
  -/
  rw [edist_comm y]; apply edist_triangle
                     /-
                       🎉 no goals
                     -/


theorem edist_congr_right {x y z : α} (h : edist x y = 0) : edist x z = edist y z := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y z : α
    h : Eq (EDist.edist x y) 0
    ⊢ Eq (EDist.edist x z) (EDist.edist y z)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x y z : α
      h : Eq (EDist.edist x y) 0
      ⊢ LE.le (EDist.edist x z) (EDist.edist y z)
    -/
  · rw [← zero_add (edist y z), ← h]
    /-
      case a
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x y z : α
      h : Eq (EDist.edist x y) 0
      ⊢ LE.le (EDist.edist x z) (HAdd.hAdd (EDist.edist x y) (EDist.edist y z))
    -/
    apply edist_triangle
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x y z : α
      h : Eq (EDist.edist x y) 0
      ⊢ LE.le (EDist.edist y z) (EDist.edist x z)
    -/
  · rw [edist_comm] at h
    /-
      case a
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x y z : α
      h : Eq (EDist.edist y x) 0
      ⊢ LE.le (EDist.edist y z) (EDist.edist x z)
    -/
    rw [← zero_add (edist x z), ← h]
    /-
      case a
      α : Type u
      inst✝ : PseudoEMetricSpace α
      x y z : α
      h : Eq (EDist.edist y x) 0
      ⊢ LE.le (EDist.edist y z) (HAdd.hAdd (EDist.edist y x) (EDist.edist x z))
    -/
    apply edist_triangle
    /-
      🎉 no goals
    -/


theorem edist_congr_left {x y z : α} (h : edist x y = 0) : edist z x = edist z y := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y z : α
    h : Eq (EDist.edist x y) 0
    ⊢ Eq (EDist.edist z x) (EDist.edist z y)
  -/
  rw [edist_comm z x, edist_comm z y]
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y z : α
    h : Eq (EDist.edist x y) 0
    ⊢ Eq (EDist.edist x z) (EDist.edist y z)
  -/
  apply edist_congr_right h
  /-
    🎉 no goals
  -/

-- new theorem

theorem edist_congr {w x y z : α} (hl : edist w x = 0) (hr : edist y z = 0) :
    edist w y = edist x z :=
  (edist_congr_right hl).trans (edist_congr_left hr)


theorem edist_triangle4 (x y z t : α) : edist x t ≤ edist x y + edist y z + edist z t :=
  calc
    edist x t ≤ edist x z + edist z t := edist_triangle x z t
    _ ≤ edist x y + edist y z + edist z t := add_le_add_right (edist_triangle x y z) _


/-- Reformulation of the uniform structure in terms of the extended distance -/
theorem uniformity_pseudoedist : 𝓤 α = ⨅ ε > 0, 𝓟 { p : α × α | edist p.1 p.2 < ε } :=
  PseudoEMetricSpace.uniformity_edist


theorem uniformSpace_edist :
    ‹PseudoEMetricSpace α›.toUniformSpace =
      uniformSpaceOfEDist edist edist_self edist_comm edist_triangle :=
  UniformSpace.ext uniformity_pseudoedist


theorem uniformity_basis_edist :
    (𝓤 α).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) fun ε => { p : α × α | edist p.1 p.2 < ε } :=
  (@uniformSpace_edist α _).symm ▸ UniformSpace.hasBasis_ofFun ⟨1, one_pos⟩ _ _ _ _ _


/-- Characterization of the elements of the uniformity in terms of the extended distance -/
theorem mem_uniformity_edist {s : Set (α × α)} :
    s ∈ 𝓤 α ↔ ∃ ε > 0, ∀ {a b : α}, edist a b < ε → (a, b) ∈ s :=
  uniformity_basis_edist.mem_uniformity_iff


/-- Given `f : β → ℝ≥0∞`, if `f` sends `{i | p i}` to a set of positive numbers
accumulating to zero, then `f i`-neighborhoods of the diagonal form a basis of `𝓤 α`.

For specific bases see `uniformity_basis_edist`, `uniformity_basis_edist'`,
`uniformity_basis_edist_nnreal`, and `uniformity_basis_edist_inv_nat`. -/
protected theorem EMetric.mk_uniformity_basis {β : Type*} {p : β → Prop} {f : β → ℝ≥0∞}
    (hf₀ : ∀ x, p x → 0 < f x) (hf : ∀ ε, 0 < ε → ∃ x, p x ∧ f x ≤ ε) :
    (𝓤 α).HasBasis p fun x => { p : α × α | edist p.1 p.2 < f x } := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    β : Type u_2
    p : β → Prop
    f : β → ENNReal
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    ⊢ (uniformity α).HasBasis p fun x => setOf fun p => LT.lt (EDist.edist p.1 p.2 …
  -/
  refine ⟨fun s => uniformity_basis_edist.mem_iff.trans ?_⟩
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    β : Type u_2
    p : β → Prop
    f : β → ENNReal
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    s : Set (Prod α α)
    ⊢ Iff (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt (ED …
    -/
  · rintro ⟨ε, ε₀, hε⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : ENNReal
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (EDist.edist p.1 p.2) ε) s
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (EDist.edi …
    -/
    rcases hf ε ε₀ with ⟨i, hi, H⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : ENNReal
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (EDist.edist p.1 p.2) ε) s
      i : β
      hi : p i
      H : LE.le (f i) ε
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (EDist.edi …
    -/
    exact ⟨i, hi, fun x hx => hε <| lt_of_lt_of_le hx.out H⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (EDist.ed …
    -/
  · exact fun ⟨i, hi, H⟩ => ⟨f i, hf₀ i hi, H⟩
    /-
      🎉 no goals
    -/


/-- Given `f : β → ℝ≥0∞`, if `f` sends `{i | p i}` to a set of positive numbers
accumulating to zero, then closed `f i`-neighborhoods of the diagonal form a basis of `𝓤 α`.

For specific bases see `uniformity_basis_edist_le` and `uniformity_basis_edist_le'`. -/
protected theorem EMetric.mk_uniformity_basis_le {β : Type*} {p : β → Prop} {f : β → ℝ≥0∞}
    (hf₀ : ∀ x, p x → 0 < f x) (hf : ∀ ε, 0 < ε → ∃ x, p x ∧ f x ≤ ε) :
    (𝓤 α).HasBasis p fun x => { p : α × α | edist p.1 p.2 ≤ f x } := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    β : Type u_2
    p : β → Prop
    f : β → ENNReal
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    ⊢ (uniformity α).HasBasis p fun x => setOf fun p => LE.le (EDist.edist p.1 p.2 …
  -/
  refine ⟨fun s => uniformity_basis_edist.mem_iff.trans ?_⟩
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    β : Type u_2
    p : β → Prop
    f : β → ENNReal
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    s : Set (Prod α α)
    ⊢ Iff (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt (ED …
    -/
  · rintro ⟨ε, ε₀, hε⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : ENNReal
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (EDist.edist p.1 p.2) ε) s
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (EDist.edi …
    -/
    rcases exists_between ε₀ with ⟨ε', hε'⟩
    /-
      case mp.intro.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : ENNReal
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (EDist.edist p.1 p.2) ε) s
      ε' : ENNReal
      hε' : And (LT.lt 0 ε') (LT.lt ε' ε)
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (EDist.edi …
    -/
    rcases hf ε' hε'.1 with ⟨i, hi, H⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : ENNReal
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (EDist.edist p.1 p.2) ε) s
      ε' : ENNReal
      hε' : And (LT.lt 0 ε') (LT.lt ε' ε)
      i : β
      hi : p i
      H : LE.le (f i) ε'
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (EDist.edi …
    -/
    exact ⟨i, hi, fun x hx => hε <| lt_of_le_of_lt (le_trans hx.out H) hε'.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : PseudoEMetricSpace α
      β : Type u_2
      p : β → Prop
      f : β → ENNReal
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : ENNReal), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (EDist.ed …
    -/
  · exact fun ⟨i, hi, H⟩ => ⟨f i, hf₀ i hi, fun x hx => H (le_of_lt hx.out)⟩
    /-
      🎉 no goals
    -/


theorem uniformity_basis_edist_le :
    (𝓤 α).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) fun ε => { p : α × α | edist p.1 p.2 ≤ ε } :=
  EMetric.mk_uniformity_basis_le (fun _ => id) fun ε ε₀ => ⟨ε, ε₀, le_refl ε⟩


theorem uniformity_basis_edist' (ε' : ℝ≥0∞) (hε' : 0 < ε') :
    (𝓤 α).HasBasis (fun ε : ℝ≥0∞ => ε ∈ Ioo 0 ε') fun ε => { p : α × α | edist p.1 p.2 < ε } :=
  EMetric.mk_uniformity_basis (fun _ => And.left) fun ε ε₀ =>
    let ⟨δ, hδ⟩ := exists_between hε'
    ⟨min ε δ, ⟨lt_min ε₀ hδ.1, lt_of_le_of_lt (min_le_right _ _) hδ.2⟩, min_le_left _ _⟩


theorem uniformity_basis_edist_le' (ε' : ℝ≥0∞) (hε' : 0 < ε') :
    (𝓤 α).HasBasis (fun ε : ℝ≥0∞ => ε ∈ Ioo 0 ε') fun ε => { p : α × α | edist p.1 p.2 ≤ ε } :=
  EMetric.mk_uniformity_basis_le (fun _ => And.left) fun ε ε₀ =>
    let ⟨δ, hδ⟩ := exists_between hε'
    ⟨min ε δ, ⟨lt_min ε₀ hδ.1, lt_of_le_of_lt (min_le_right _ _) hδ.2⟩, min_le_left _ _⟩


theorem uniformity_basis_edist_nnreal :
    (𝓤 α).HasBasis (fun ε : ℝ≥0 => 0 < ε) fun ε => { p : α × α | edist p.1 p.2 < ε } :=
  EMetric.mk_uniformity_basis (fun _ => ENNReal.coe_pos.2) fun _ε ε₀ =>
    let ⟨δ, hδ⟩ := ENNReal.lt_iff_exists_nnreal_btwn.1 ε₀
    ⟨δ, ENNReal.coe_pos.1 hδ.1, le_of_lt hδ.2⟩


theorem uniformity_basis_edist_nnreal_le :
    (𝓤 α).HasBasis (fun ε : ℝ≥0 => 0 < ε) fun ε => { p : α × α | edist p.1 p.2 ≤ ε } :=
  EMetric.mk_uniformity_basis_le (fun _ => ENNReal.coe_pos.2) fun _ε ε₀ =>
    let ⟨δ, hδ⟩ := ENNReal.lt_iff_exists_nnreal_btwn.1 ε₀
    ⟨δ, ENNReal.coe_pos.1 hδ.1, le_of_lt hδ.2⟩


theorem uniformity_basis_edist_inv_nat :
    (𝓤 α).HasBasis (fun _ => True) fun n : ℕ => { p : α × α | edist p.1 p.2 < (↑n)⁻¹ } :=
  EMetric.mk_uniformity_basis (fun n _ ↦ ENNReal.inv_pos.2 <| ENNReal.natCast_ne_top n) fun _ε ε₀ ↦
    let ⟨n, hn⟩ := ENNReal.exists_inv_nat_lt (ne_of_gt ε₀)
    ⟨n, trivial, le_of_lt hn⟩


theorem uniformity_basis_edist_inv_two_pow :
    (𝓤 α).HasBasis (fun _ => True) fun n : ℕ => { p : α × α | edist p.1 p.2 < 2⁻¹ ^ n } :=
  EMetric.mk_uniformity_basis (fun _ _ => ENNReal.pow_pos (ENNReal.inv_pos.2 ENNReal.two_ne_top) _)
    fun _ε ε₀ =>
    let ⟨n, hn⟩ := ENNReal.exists_inv_two_pow_lt (ne_of_gt ε₀)
    ⟨n, trivial, le_of_lt hn⟩


/-- Fixed size neighborhoods of the diagonal belong to the uniform structure -/
theorem edist_mem_uniformity {ε : ℝ≥0∞} (ε0 : 0 < ε) : { p : α × α | edist p.1 p.2 < ε } ∈ 𝓤 α :=
  mem_uniformity_edist.2 ⟨ε, ε0, id⟩


instance (priority := 900) instIsCountablyGeneratedUniformity : IsCountablyGenerated (𝓤 α) :=
  isCountablyGenerated_of_seq ⟨_, uniformity_basis_edist_inv_nat.eq_iInf⟩

-- Porting note: changed explicit/implicit

/-- ε-δ characterization of uniform continuity on a set for pseudoemetric spaces -/
theorem uniformContinuousOn_iff [PseudoEMetricSpace β] {f : α → β} {s : Set α} :
    UniformContinuousOn f s ↔
      ∀ ε > 0, ∃ δ > 0, ∀ {a}, a ∈ s → ∀ {b}, b ∈ s → edist a b < δ → edist (f a) (f b) < ε :=
  uniformity_basis_edist.uniformContinuousOn_iff uniformity_basis_edist


/-- ε-δ characterization of uniform continuity on pseudoemetric spaces -/
theorem uniformContinuous_iff [PseudoEMetricSpace β] {f : α → β} :
    UniformContinuous f ↔ ∀ ε > 0, ∃ δ > 0, ∀ {a b : α}, edist a b < δ → edist (f a) (f b) < ε :=
  uniformity_basis_edist.uniformContinuous_iff uniformity_basis_edist


/-- Auxiliary function to replace the uniformity on a pseudoemetric space with
a uniformity which is equal to the original one, but maybe not defeq.
This is useful if one wants to construct a pseudoemetric space with a
specified uniformity. See Note [forgetful inheritance] explaining why having definitionally
the right uniformity is often important.
See note [reducible non-instances].
-/
abbrev PseudoEMetricSpace.replaceUniformity {α} [U : UniformSpace α] (m : PseudoEMetricSpace α)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : PseudoEMetricSpace α where
  edist := @edist _ m.toEDist
  edist_self := edist_self
  edist_comm := edist_comm
  edist_triangle := edist_triangle
  toUniformSpace := U
  uniformity_edist := H.trans (@PseudoEMetricSpace.uniformity_edist α _)


/-- The extended pseudometric induced by a function taking values in a pseudoemetric space.
See note [reducible non-instances]. -/
abbrev PseudoEMetricSpace.induced {α β} (f : α → β) (m : PseudoEMetricSpace β) :
    PseudoEMetricSpace α where
  edist x y := edist (f x) (f y)
  edist_self _ := edist_self _
  edist_comm _ _ := edist_comm _ _
  edist_triangle _ _ _ := edist_triangle _ _ _
  toUniformSpace := UniformSpace.comap f m.toUniformSpace
  uniformity_edist := (uniformity_basis_edist.comap (Prod.map f f)).eq_biInf


/-- Pseudoemetric space instance on subsets of pseudoemetric spaces -/
instance {α : Type*} {p : α → Prop} [PseudoEMetricSpace α] : PseudoEMetricSpace (Subtype p) :=
  PseudoEMetricSpace.induced Subtype.val ‹_›


/-- The extended pseudodistance on a subset of a pseudoemetric space is the restriction of
the original pseudodistance, by definition. -/
theorem Subtype.edist_eq {p : α → Prop} (x y : Subtype p) : edist x y = edist (x : α) y := rfl


/-- The extended pseudodistance on a subtype of a pseudoemetric space is the restriction of
the original pseudodistance, by definition. -/
@[simp]
theorem Subtype.edist_mk_mk {p : α → Prop} {x y : α} (hx : p x) (hy : p y) :
    edist (⟨x, hx⟩ : Subtype p) ⟨y, hy⟩ = edist x y :=
  rfl


/-- Pseudoemetric space instance on the multiplicative opposite of a pseudoemetric space. -/
@[to_additive "Pseudoemetric space instance on the additive opposite of a pseudoemetric space."]
instance {α : Type*} [PseudoEMetricSpace α] : PseudoEMetricSpace αᵐᵒᵖ :=
  PseudoEMetricSpace.induced unop ‹_›


@[to_additive]
theorem edist_unop (x y : αᵐᵒᵖ) : edist (unop x) (unop y) = edist x y := rfl


@[to_additive]
theorem edist_op (x y : α) : edist (op x) (op y) = edist x y := rfl


instance : PseudoEMetricSpace (ULift α) := PseudoEMetricSpace.induced ULift.down ‹_›


theorem ULift.edist_eq (x y : ULift α) : edist x y = edist x.down y.down := rfl


@[simp]
theorem ULift.edist_up_up (x y : α) : edist (ULift.up x) (ULift.up y) = edist x y := rfl


/-- The product of two pseudoemetric spaces, with the max distance, is an extended
pseudometric spaces. We make sure that the uniform structure thus constructed is the one
corresponding to the product of uniform spaces, to avoid diamond problems. -/
instance Prod.pseudoEMetricSpaceMax [PseudoEMetricSpace β] :
  PseudoEMetricSpace (α × β) where
  edist x y := edist x.1 y.1 ⊔ edist x.2 y.2
                     /-
                       α : Type u
                       β : Type v
                       X : Type u_1
                       inst✝¹ : PseudoEMetricSpace α
                       inst✝ : PseudoEMetricSpace β
                       x : Prod α β
                       ⊢ Eq (EDist.edist x x) 0
                     -/
  edist_self x := by simp
                     /-
                       🎉 no goals
                     -/
                       /-
                         α : Type u
                         β : Type v
                         X : Type u_1
                         inst✝¹ : PseudoEMetricSpace α
                         inst✝ : PseudoEMetricSpace β
                         x y : Prod α β
                         ⊢ Eq (EDist.edist x y) (EDist.edist y x)
                       -/
  edist_comm x y := by simp [edist_comm]
                       /-
                         🎉 no goals
                       -/
  edist_triangle _ _ _ :=
    max_le (le_trans (edist_triangle _ _ _) (add_le_add (le_max_left _ _) (le_max_left _ _)))
      (le_trans (edist_triangle _ _ _) (add_le_add (le_max_right _ _) (le_max_right _ _)))
  uniformity_edist := uniformity_prod.trans <| by
    /-
      α : Type u
      β : Type v
      X : Type u_1
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      ⊢ Eq (Min.min (Filter.comap (fun p => { fst := p.1.1, snd := p.2.1 }) (uniform …
    -/
    simp [PseudoEMetricSpace.uniformity_edist, ← iInf_inf_eq, setOf_and]
    /-
      🎉 no goals
    -/
  toUniformSpace := inferInstance


theorem Prod.edist_eq [PseudoEMetricSpace β] (x y : α × β) :
    edist x y = max (edist x.1 y.1) (edist x.2 y.2) :=
  rfl


/-- `EMetric.ball x ε` is the set of all points `y` with `edist y x < ε` -/
def ball (x : α) (ε : ℝ≥0∞) : Set α :=
  { y | edist y x < ε }


@[simp] theorem mem_ball : y ∈ ball x ε ↔ edist y x < ε := Iff.rfl


                                                       /-
                                                         α : Type u
                                                         inst✝ : PseudoEMetricSpace α
                                                         x y : α
                                                         ε : ENNReal
                                                         ⊢ Iff (Membership.mem (EMetric.ball x ε) y) (LT.lt (EDist.edist x y) ε)
                                                       -/
theorem mem_ball' : y ∈ ball x ε ↔ edist x y < ε := by rw [edist_comm, mem_ball]
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- `EMetric.closedBall x ε` is the set of all points `y` with `edist y x ≤ ε` -/
def closedBall (x : α) (ε : ℝ≥0∞) :=
  { y | edist y x ≤ ε }


@[simp] theorem mem_closedBall : y ∈ closedBall x ε ↔ edist y x ≤ ε := Iff.rfl


                                                                   /-
                                                                     α : Type u
                                                                     inst✝ : PseudoEMetricSpace α
                                                                     x y : α
                                                                     ε : ENNReal
                                                                     ⊢ Iff (Membership.mem (EMetric.closedBall x ε) y) (LE.le (EDist.edist x y) ε)
                                                                   -/
theorem mem_closedBall' : y ∈ closedBall x ε ↔ edist x y ≤ ε := by rw [edist_comm, mem_closedBall]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem closedBall_top (x : α) : closedBall x ∞ = univ :=
  eq_univ_of_forall fun _ => mem_setOf.2 le_top


theorem ball_subset_closedBall : ball x ε ⊆ closedBall x ε := fun _ h => le_of_lt h.out


theorem pos_of_mem_ball (hy : y ∈ ball x ε) : 0 < ε :=
  lt_of_le_of_lt (zero_le _) hy


theorem mem_ball_self (h : 0 < ε) : x ∈ ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    ε : ENNReal
    h : LT.lt 0 ε
    ⊢ Membership.mem (EMetric.ball x ε) x
  -/
  rwa [mem_ball, edist_self]
  /-
    🎉 no goals
  -/


theorem mem_closedBall_self : x ∈ closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x : α
    ε : ENNReal
    ⊢ Membership.mem (EMetric.closedBall x ε) x
  -/
  rw [mem_closedBall, edist_self]; apply zero_le
                                   /-
                                     🎉 no goals
                                   -/


                                                          /-
                                                            α : Type u
                                                            inst✝ : PseudoEMetricSpace α
                                                            x y : α
                                                            ε : ENNReal
                                                            ⊢ Iff (Membership.mem (EMetric.ball y ε) x) (Membership.mem (EMetric.ball x ε) …
                                                          -/
theorem mem_ball_comm : x ∈ ball y ε ↔ y ∈ ball x ε := by rw [mem_ball', mem_ball]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem mem_closedBall_comm : x ∈ closedBall y ε ↔ y ∈ closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    ε : ENNReal
    ⊢ Iff (Membership.mem (EMetric.closedBall y ε) x) (Membership.mem (EMetric.clo …
  -/
  rw [mem_closedBall', mem_closedBall]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem ball_subset_ball (h : ε₁ ≤ ε₂) : ball x ε₁ ⊆ ball x ε₂ := fun _y (yx : _ < ε₁) =>
  lt_of_lt_of_le yx h


@[gcongr]
theorem closedBall_subset_closedBall (h : ε₁ ≤ ε₂) : closedBall x ε₁ ⊆ closedBall x ε₂ :=
  fun _y (yx : _ ≤ ε₁) => le_trans yx h


theorem ball_disjoint (h : ε₁ + ε₂ ≤ edist x y) : Disjoint (ball x ε₁) (ball y ε₂) :=
  Set.disjoint_left.mpr fun z h₁ h₂ =>
    (edist_triangle_left x y z).not_lt <| (ENNReal.add_lt_add h₁ h₂).trans_le h


theorem ball_subset (h : edist x y + ε₁ ≤ ε₂) (h' : edist x y ≠ ∞) : ball x ε₁ ⊆ ball y ε₂ :=
  fun z zx =>
  calc
    edist z y ≤ edist z x + edist x y := edist_triangle _ _ _
    _ = edist x y + edist z x := add_comm _ _
    _ < edist x y + ε₁ := ENNReal.add_lt_add_left h' zx
    _ ≤ ε₂ := h


theorem exists_ball_subset_ball (h : y ∈ ball x ε) : ∃ ε' > 0, ball y ε' ⊆ ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    ε : ENNReal
    h : Membership.mem (EMetric.ball x ε) y
    ⊢ Exists fun ε' => And (GT.gt ε' 0) (HasSubset.Subset (EMetric.ball y ε') (EMe …
  -/
  have : 0 < ε - edist y x := by simpa using h
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    ε : ENNReal
    h : Membership.mem (EMetric.ball x ε) y
    this : LT.lt 0 (HSub.hSub ε (EDist.edist y x))
    ⊢ Exists fun ε' => And (GT.gt ε' 0) (HasSubset.Subset (EMetric.ball y ε') (EMe …
  -/
  refine ⟨ε - edist y x, this, ball_subset ?_ (ne_top_of_lt h)⟩
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    x y : α
    ε : ENNReal
    h : Membership.mem (EMetric.ball x ε) y
    this : LT.lt 0 (HSub.hSub ε (EDist.edist y x))
    ⊢ LE.le (HAdd.hAdd (EDist.edist y x) (HSub.hSub ε (EDist.edist y x))) ε
  -/
  exact (add_tsub_cancel_of_le (mem_ball.mp h).le).le
  /-
    🎉 no goals
  -/


theorem ball_eq_empty_iff : ball x ε = ∅ ↔ ε = 0 :=
  eq_empty_iff_forall_not_mem.trans
    ⟨fun h => le_bot_iff.1 (le_of_not_gt fun ε0 => h _ (mem_ball_self ε0)), fun ε0 _ h =>
      not_lt_of_le (le_of_eq ε0) (pos_of_mem_ball h)⟩


theorem ordConnected_setOf_closedBall_subset (x : α) (s : Set α) :
    OrdConnected { r | closedBall x r ⊆ s } :=
  ⟨fun _ _ _ h₁ _ h₂ => (closedBall_subset_closedBall h₂.2).trans h₁⟩


theorem ordConnected_setOf_ball_subset (x : α) (s : Set α) : OrdConnected { r | ball x r ⊆ s } :=
  ⟨fun _ _ _ h₁ _ h₂ => (ball_subset_ball h₂.2).trans h₁⟩


/-- Relation “two points are at a finite edistance” is an equivalence relation. -/
def edistLtTopSetoid : Setoid α where
  r x y := edist x y < ⊤
  iseqv :=
                 /-
                   α : Type u
                   β : Type v
                   X : Type u_1
                   inst✝ : PseudoEMetricSpace α
                   x✝ y z : α
                   ε ε₁ ε₂ : ENNReal
                   s t : Set α
                   x : α
                   ⊢ LT.lt (EDist.edist x x) Top.top
                 -/
    ⟨fun x => by rw [edist_self]; exact ENNReal.coe_lt_top,
                                  /-
                                    🎉 no goals
                                  -/
                  /-
                    α : Type u
                    β : Type v
                    X : Type u_1
                    inst✝ : PseudoEMetricSpace α
                    x y z : α
                    ε ε₁ ε₂ : ENNReal
                    s t : Set α
                    x✝ y✝ : α
                    h : LT.lt (EDist.edist x✝ y✝) Top.top
                    ⊢ LT.lt (EDist.edist y✝ x✝) Top.top
                  -/
      fun h => by rwa [edist_comm], fun hxy hyz =>
                  /-
                    🎉 no goals
                  -/
        lt_of_le_of_lt (edist_triangle _ _ _) (ENNReal.add_lt_top.2 ⟨hxy, hyz⟩)⟩


@[simp]
                                       /-
                                         α : Type u
                                         inst✝ : PseudoEMetricSpace α
                                         x : α
                                         ⊢ Eq (EMetric.ball x 0) EmptyCollection.emptyCollection
                                       -/
theorem ball_zero : ball x 0 = ∅ := by rw [EMetric.ball_eq_empty_iff]
                                       /-
                                         🎉 no goals
                                       -/


theorem nhds_basis_eball : (𝓝 x).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) (ball x) :=
  nhds_basis_uniformity uniformity_basis_edist


theorem nhdsWithin_basis_eball : (𝓝[s] x).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) fun ε => ball x ε ∩ s :=
  nhdsWithin_hasBasis nhds_basis_eball s


theorem nhds_basis_closed_eball : (𝓝 x).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) (closedBall x) :=
  nhds_basis_uniformity uniformity_basis_edist_le


theorem nhdsWithin_basis_closed_eball :
    (𝓝[s] x).HasBasis (fun ε : ℝ≥0∞ => 0 < ε) fun ε => closedBall x ε ∩ s :=
  nhdsWithin_hasBasis nhds_basis_closed_eball s


theorem nhds_eq : 𝓝 x = ⨅ ε > 0, 𝓟 (ball x ε) :=
  nhds_basis_eball.eq_biInf


theorem mem_nhds_iff : s ∈ 𝓝 x ↔ ∃ ε > 0, ball x ε ⊆ s :=
  nhds_basis_eball.mem_iff


theorem mem_nhdsWithin_iff : s ∈ 𝓝[t] x ↔ ∃ ε > 0, ball x ε ∩ t ⊆ s :=
  nhdsWithin_basis_eball.mem_iff


theorem tendsto_nhdsWithin_nhdsWithin {t : Set β} {a b} :
    Tendsto f (𝓝[s] a) (𝓝[t] b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ ⦃x⦄, x ∈ s → edist x a < δ → f x ∈ t ∧ edist (f x) b < ε :=
  (nhdsWithin_basis_eball.tendsto_iff nhdsWithin_basis_eball).trans <|
    forall₂_congr fun ε _ => exists_congr fun δ => and_congr_right fun _ =>
                                /-
                                  α : Type u
                                  β : Type v
                                  inst✝¹ : PseudoEMetricSpace α
                                  s : Set α
                                  inst✝ : PseudoEMetricSpace β
                                  f : α → β
                                  t : Set β
                                  a : α
                                  b : β
                                  ε : ENNReal
                                  x✝¹ : LT.lt 0 ε
                                  δ : ENNReal
                                  x✝ : LT.lt 0 δ
                                  x : α
                                  ⊢ Iff (Membership.mem (Inter.inter (EMetric.ball a δ) s) x → Membership.mem (I …
                                -/
      forall_congr' fun x => by simp; tauto
                                      /-
                                        🎉 no goals
                                      -/


theorem tendsto_nhdsWithin_nhds {a b} :
    Tendsto f (𝓝[s] a) (𝓝 b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ {x : α}, x ∈ s → edist x a < δ → edist (f x) b < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    a : α
    b : β
    ⊢ Iff (Filter.Tendsto f (nhdsWithin a s) (nhds b)) (∀ (ε : ENNReal), GT.gt ε 0 …
  -/
  rw [← nhdsWithin_univ b, tendsto_nhdsWithin_nhdsWithin]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoEMetricSpace α
    s : Set α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    a : α
    b : β
    ⊢ Iff (∀ (ε : ENNReal), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : α⦄ …
  -/
  simp only [mem_univ, true_and]
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_nhds {a b} :
    Tendsto f (𝓝 a) (𝓝 b) ↔ ∀ ε > 0, ∃ δ > 0, ∀ ⦃x⦄, edist x a < δ → edist (f x) b < ε :=
  nhds_basis_eball.tendsto_iff nhds_basis_eball


theorem isOpen_iff : IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ball x ε ⊆ s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    ⊢ Iff (IsOpen s) (∀ (x : α), Membership.mem s x → Exists fun ε => And (GT.gt ε …
  -/
  simp [isOpen_iff_nhds, mem_nhds_iff]
  /-
    🎉 no goals
  -/


theorem isOpen_ball : IsOpen (ball x ε) :=
  isOpen_iff.2 fun _ => exists_ball_subset_ball


theorem isClosed_ball_top : IsClosed (ball x ⊤) :=
  isOpen_compl_iff.1 <| isOpen_iff.2 fun _y hy =>
    ⟨⊤, ENNReal.coe_lt_top, fun _z hzy hzx =>
      hy (edistLtTopSetoid.trans (edistLtTopSetoid.symm hzy) hzx)⟩


theorem ball_mem_nhds (x : α) {ε : ℝ≥0∞} (ε0 : 0 < ε) : ball x ε ∈ 𝓝 x :=
  isOpen_ball.mem_nhds (mem_ball_self ε0)


theorem closedBall_mem_nhds (x : α) {ε : ℝ≥0∞} (ε0 : 0 < ε) : closedBall x ε ∈ 𝓝 x :=
  mem_of_superset (ball_mem_nhds x ε0) ball_subset_closedBall


theorem ball_prod_same [PseudoEMetricSpace β] (x : α) (y : β) (r : ℝ≥0∞) :
    ball x r ×ˢ ball y r = ball (x, y) r :=
                  /-
                    α : Type u
                    β : Type v
                    inst✝¹ : PseudoEMetricSpace α
                    inst✝ : PseudoEMetricSpace β
                    x : α
                    y : β
                    r : ENNReal
                    z : Prod α β
                    ⊢ Iff (Membership.mem (SProd.sprod (EMetric.ball x r) (EMetric.ball y r)) z) ( …
                  -/
  ext fun z => by simp [Prod.edist_eq]
                  /-
                    🎉 no goals
                  -/


theorem closedBall_prod_same [PseudoEMetricSpace β] (x : α) (y : β) (r : ℝ≥0∞) :
    closedBall x r ×ˢ closedBall y r = closedBall (x, y) r :=
                  /-
                    α : Type u
                    β : Type v
                    inst✝¹ : PseudoEMetricSpace α
                    inst✝ : PseudoEMetricSpace β
                    x : α
                    y : β
                    r : ENNReal
                    z : Prod α β
                    ⊢ Iff (Membership.mem (SProd.sprod (EMetric.closedBall x r) (EMetric.closedBal …
                  -/
  ext fun z => by simp [Prod.edist_eq]
                  /-
                    🎉 no goals
                  -/


/-- ε-characterization of the closure in pseudoemetric spaces -/
theorem mem_closure_iff : x ∈ closure s ↔ ∀ ε > 0, ∃ y ∈ s, edist x y < ε :=
                                                            /-
                                                              α : Type u
                                                              inst✝ : PseudoEMetricSpace α
                                                              x : α
                                                              s : Set α
                                                              ⊢ Iff (∀ (i : ENNReal), LT.lt 0 i → Exists fun y => And (Membership.mem s y) ( …
                                                            -/
  (mem_closure_iff_nhds_basis nhds_basis_eball).trans <| by simp only [mem_ball, edist_comm x]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem tendsto_nhds {f : Filter β} {u : β → α} {a : α} :
    Tendsto u f (𝓝 a) ↔ ∀ ε > 0, ∀ᶠ x in f, edist (u x) a < ε :=
  nhds_basis_eball.tendsto_right_iff


theorem tendsto_atTop [Nonempty β] [SemilatticeSup β] {u : β → α} {a : α} :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, edist (u n) a < ε :=
  (atTop_basis.tendsto_iff nhds_basis_eball).trans <| by
    /-
      α : Type u
      β : Type v
      inst✝² : PseudoEMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      a : α
      ⊢ Iff (∀ (ib : ENNReal), LT.lt 0 ib → Exists fun ia => And True (∀ (x : β), Me …
    -/
    simp only [exists_prop, true_and, mem_Ici, mem_ball]
    /-
      🎉 no goals
    -/


/-- For a set `s` in a pseudo emetric space, if for every `ε > 0` there exists a countable
set that is `ε`-dense in `s`, then there exists a countable subset `t ⊆ s` that is dense in `s`. -/
theorem subset_countable_closure_of_almost_dense_set (s : Set α)
    (hs : ∀ ε > 0, ∃ t : Set α, t.Countable ∧ s ⊆ ⋃ x ∈ t, closedBall x ε) :
    ∃ t, t ⊆ s ∧ t.Countable ∧ s ⊆ closure t := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subse …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | ⟨x₀, hx₀⟩)
    /-
      case inl
      α : Type u
      inst✝ : PseudoEMetricSpace α
      hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
      ⊢ Exists fun t => And (HasSubset.Subset t EmptyCollection.emptyCollection) (An …
    -/
  · exact ⟨∅, empty_subset _, countable_empty, empty_subset _⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subse …
  -/
  choose! T hTc hsT using fun n : ℕ => hs n⁻¹ (by simp)
  have : ∀ r x, ∃ y ∈ s, closedBall x r ∩ s ⊆ closedBall y (r * 2) := fun r x => by
    rcases (closedBall x r ∩ s).eq_empty_or_nonempty with (he | ⟨y, hxy, hys⟩)
    · refine ⟨x₀, hx₀, ?_⟩
      rw [he]
      exact empty_subset _
    · refine ⟨y, hys, fun z hz => ?_⟩
      calc
        edist z y ≤ edist z x + edist y x := edist_triangle_right _ _ _
        _ ≤ r + r := add_le_add hz.1 hxy
        _ = r * 2 := (mul_two r).symm
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    T : Nat → Set α
    hTc : ∀ (n : Nat), (T n).Countable
    hsT : ∀ (n : Nat), HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => …
    this : ∀ (r : ENNReal) (x : α), Exists fun y => And (Membership.mem s y) (HasS …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subse …
  -/
  choose f hfs hf using this
  refine
    ⟨⋃ n : ℕ, f n⁻¹ '' T n, iUnion_subset fun n => image_subset_iff.2 fun z _ => hfs _ _,
      countable_iUnion fun n => (hTc n).image _, ?_⟩
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    T : Nat → Set α
    hTc : ∀ (n : Nat), (T n).Countable
    hsT : ∀ (n : Nat), HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => …
    f : ENNReal → α → α
    hfs : ∀ (r : ENNReal) (x : α), Membership.mem s (f r x)
    hf : ∀ (r : ENNReal) (x : α), HasSubset.Subset (Inter.inter (EMetric.closedBal …
    ⊢ HasSubset.Subset s (closure (Set.iUnion fun n => Set.image (f (Inv.inv ↑n))  …
  -/
  refine fun x hx => mem_closure_iff.2 fun ε ε0 => ?_
  /-
    case inr.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    T : Nat → Set α
    hTc : ∀ (n : Nat), (T n).Countable
    hsT : ∀ (n : Nat), HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => …
    f : ENNReal → α → α
    hfs : ∀ (r : ENNReal) (x : α), Membership.mem s (f r x)
    hf : ∀ (r : ENNReal) (x : α), HasSubset.Subset (Inter.inter (EMetric.closedBal …
    x : α
    hx : Membership.mem s x
    ε : ENNReal
    ε0 : GT.gt ε 0
    ⊢ Exists fun y => And (Membership.mem (Set.iUnion fun n => Set.image (f (Inv.i …
  -/
  rcases ENNReal.exists_inv_nat_lt (ENNReal.half_pos ε0.lt.ne').ne' with ⟨n, hn⟩
  /-
    case inr.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    T : Nat → Set α
    hTc : ∀ (n : Nat), (T n).Countable
    hsT : ∀ (n : Nat), HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => …
    f : ENNReal → α → α
    hfs : ∀ (r : ENNReal) (x : α), Membership.mem s (f r x)
    hf : ∀ (r : ENNReal) (x : α), HasSubset.Subset (Inter.inter (EMetric.closedBal …
    x : α
    hx : Membership.mem s x
    ε : ENNReal
    ε0 : GT.gt ε 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) (HDiv.hDiv ε 2)
    ⊢ Exists fun y => And (Membership.mem (Set.iUnion fun n => Set.image (f (Inv.i …
  -/
  rcases mem_iUnion₂.1 (hsT n hx) with ⟨y, hyn, hyx⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset.S …
    x₀ : α
    hx₀ : Membership.mem s x₀
    T : Nat → Set α
    hTc : ∀ (n : Nat), (T n).Countable
    hsT : ∀ (n : Nat), HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => …
    f : ENNReal → α → α
    hfs : ∀ (r : ENNReal) (x : α), Membership.mem s (f r x)
    hf : ∀ (r : ENNReal) (x : α), HasSubset.Subset (Inter.inter (EMetric.closedBal …
    x : α
    hx : Membership.mem s x
    ε : ENNReal
    ε0 : GT.gt ε 0
    n : Nat
    hn : LT.lt (Inv.inv ↑n) (HDiv.hDiv ε 2)
    y : α
    hyn : Membership.mem (T n) y
    hyx : Membership.mem (EMetric.closedBall y (Inv.inv ↑n)) x
    ⊢ Exists fun y => And (Membership.mem (Set.iUnion fun n => Set.image (f (Inv.i …
  -/
  refine ⟨f n⁻¹ y, mem_iUnion.2 ⟨n, mem_image_of_mem _ hyn⟩, ?_⟩
  calc
    edist x (f n⁻¹ y) ≤ (n : ℝ≥0∞)⁻¹ * 2 := hf _ _ ⟨hyx, hx⟩
    _ < ε := ENNReal.mul_lt_of_lt_div hn


open TopologicalSpace in
/-- If a set `s` is separable in a (pseudo extended) metric space, then it admits a countable dense
subset. This is not obvious, as the countable set whose closure covers `s` given by the definition
of separability does not need in general to be contained in `s`. -/
theorem _root_.TopologicalSpace.IsSeparable.exists_countable_dense_subset
    {s : Set α} (hs : IsSeparable s) : ∃ t, t ⊆ s ∧ t.Countable ∧ s ⊆ closure t := by
  have : ∀ ε > 0, ∃ t : Set α, t.Countable ∧ s ⊆ ⋃ x ∈ t, closedBall x ε := fun ε ε0 => by
    rcases hs with ⟨t, htc, hst⟩
    refine ⟨t, htc, hst.trans fun x hx => ?_⟩
    rcases mem_closure_iff.1 hx ε ε0 with ⟨y, hyt, hxy⟩
    exact mem_iUnion₂.2 ⟨y, hyt, mem_closedBall.2 hxy.le⟩
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    this : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun t => And t.Countable (HasSubset …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Countable (HasSubset.Subse …
  -/
  exact subset_countable_closure_of_almost_dense_set _ this
  /-
    🎉 no goals
  -/


open TopologicalSpace in
/-- If a set `s` is separable, then the corresponding subtype is separable in a (pseudo extended)
metric space.  This is not obvious, as the countable set whose closure covers `s` does not need in
general to be contained in `s`. -/
theorem _root_.TopologicalSpace.IsSeparable.separableSpace {s : Set α} (hs : IsSeparable s) :
    SeparableSpace s := by
  /-
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.SeparableSpace ↑s
  -/
  rcases hs.exists_countable_dense_subset with ⟨t, hts, htc, hst⟩
  /-
    case intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    t : Set α
    hts : HasSubset.Subset t s
    htc : t.Countable
    hst : HasSubset.Subset s (closure t)
    ⊢ TopologicalSpace.SeparableSpace ↑s
  -/
  lift t to Set s using hts
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    t : Set ↑s
    htc : (Set.image Subtype.val t).Countable
    hst : HasSubset.Subset s (closure (Set.image Subtype.val t))
    ⊢ TopologicalSpace.SeparableSpace ↑s
  -/
  refine ⟨⟨t, countable_of_injective_of_countable_image Subtype.coe_injective.injOn htc, ?_⟩⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : PseudoEMetricSpace α
    s : Set α
    hs : TopologicalSpace.IsSeparable s
    t : Set ↑s
    htc : (Set.image Subtype.val t).Countable
    hst : HasSubset.Subset s (closure (Set.image Subtype.val t))
    ⊢ Dense t
  -/
  rwa [IsInducing.subtypeVal.dense_iff, Subtype.forall]
  /-
    🎉 no goals
  -/


/-- We now define `EMetricSpace`, extending `PseudoEMetricSpace`. -/
class EMetricSpace (α : Type u) extends PseudoEMetricSpace α : Type u where
  eq_of_edist_eq_zero : ∀ {x y : α}, edist x y = 0 → x = y


@[ext]
protected theorem EMetricSpace.ext
    {α : Type*} {m m' : EMetricSpace α} (h : m.toEDist = m'.toEDist) : m = m' := by
  /-
    α : Type u_2
    m m' : EMetricSpace α
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq m m'
  -/
  cases m
  /-
    case mk
    α : Type u_2
    m' : EMetricSpace α
    toPseudoEMetricSpace✝ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq (EMetricSpace.mk eq_of_edist_eq_zero✝) m'
  -/
  cases m'
  /-
    case mk.mk
    α : Type u_2
    toPseudoEMetricSpace✝¹ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝¹ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    toPseudoEMetricSpace✝ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq (EMetricSpace.mk eq_of_edist_eq_zero✝¹) (EMetricSpace.mk eq_of_edist_eq_z …
  -/
  congr
  /-
    case mk.mk.e_toPseudoEMetricSpace
    α : Type u_2
    toPseudoEMetricSpace✝¹ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝¹ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    toPseudoEMetricSpace✝ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq toPseudoEMetricSpace✝¹ toPseudoEMetricSpace✝
  -/
  ext1
  /-
    case mk.mk.e_toPseudoEMetricSpace.h
    α : Type u_2
    toPseudoEMetricSpace✝¹ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝¹ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    toPseudoEMetricSpace✝ : PseudoEMetricSpace α
    eq_of_edist_eq_zero✝ : ∀ {x y : α}, Eq (EDist.edist x y) 0 → Eq x y
    h : Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
    ⊢ Eq PseudoEMetricSpace.toEDist PseudoEMetricSpace.toEDist
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- Characterize the equality of points by the vanishing of their extended distance -/
@[simp]
theorem edist_eq_zero {x y : γ} : edist x y = 0 ↔ x = y :=
  ⟨eq_of_edist_eq_zero, fun h => h ▸ edist_self _⟩


@[simp]
theorem zero_eq_edist {x y : γ} : 0 = edist x y ↔ x = y := eq_comm.trans edist_eq_zero


theorem edist_le_zero {x y : γ} : edist x y ≤ 0 ↔ x = y :=
  nonpos_iff_eq_zero.trans edist_eq_zero


@[simp]
                                                          /-
                                                            γ : Type w
                                                            inst✝ : EMetricSpace γ
                                                            x y : γ
                                                            ⊢ Iff (LT.lt 0 (EDist.edist x y)) (Ne x y)
                                                          -/
theorem edist_pos {x y : γ} : 0 < edist x y ↔ x ≠ y := by simp [← not_le]
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Two points coincide if their distance is `< ε` for all positive ε -/
theorem eq_of_forall_edist_le {x y : γ} (h : ∀ ε > 0, edist x y ≤ ε) : x = y :=
  eq_of_edist_eq_zero (eq_of_le_of_forall_le_of_dense bot_le h)


/-- Auxiliary function to replace the uniformity on an emetric space with
a uniformity which is equal to the original one, but maybe not defeq.
This is useful if one wants to construct an emetric space with a
specified uniformity. See Note [forgetful inheritance] explaining why having definitionally
the right uniformity is often important.
See note [reducible non-instances].
-/
abbrev EMetricSpace.replaceUniformity {γ} [U : UniformSpace γ] (m : EMetricSpace γ)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : EMetricSpace γ where
  edist := @edist _ m.toEDist
  edist_self := edist_self
  eq_of_edist_eq_zero := @eq_of_edist_eq_zero _ _
  edist_comm := edist_comm
  edist_triangle := edist_triangle
  toUniformSpace := U
  uniformity_edist := H.trans (@PseudoEMetricSpace.uniformity_edist γ _)


/-- The extended metric induced by an injective function taking values in an emetric space.
See Note [reducible non-instances]. -/
abbrev EMetricSpace.induced {γ β} (f : γ → β) (hf : Function.Injective f) (m : EMetricSpace β) :
    EMetricSpace γ :=
  { PseudoEMetricSpace.induced f m.toPseudoEMetricSpace with
    eq_of_edist_eq_zero := fun h => hf (edist_eq_zero.1 h) }


/-- EMetric space instance on subsets of emetric spaces -/
instance {α : Type*} {p : α → Prop} [EMetricSpace α] : EMetricSpace (Subtype p) :=
  EMetricSpace.induced Subtype.val Subtype.coe_injective ‹_›


/-- EMetric space instance on the multiplicative opposite of an emetric space. -/
@[to_additive "EMetric space instance on the additive opposite of an emetric space."]
instance {α : Type*} [EMetricSpace α] : EMetricSpace αᵐᵒᵖ :=
  EMetricSpace.induced MulOpposite.unop MulOpposite.unop_injective ‹_›


instance {α : Type*} [EMetricSpace α] : EMetricSpace (ULift α) :=
  EMetricSpace.induced ULift.down ULift.down_injective ‹_›


/-- Reformulation of the uniform structure in terms of the extended distance -/
theorem uniformity_edist : 𝓤 γ = ⨅ ε > 0, 𝓟 { p : γ × γ | edist p.1 p.2 < ε } :=
  PseudoEMetricSpace.uniformity_edist


instance : EDist (Additive X) := ‹EDist X›

instance : EDist (Multiplicative X) := ‹EDist X›


@[simp]
theorem edist_ofMul (a b : X) : edist (ofMul a) (ofMul b) = edist a b :=
  rfl


@[simp]
theorem edist_ofAdd (a b : X) : edist (ofAdd a) (ofAdd b) = edist a b :=
  rfl


@[simp]
theorem edist_toMul (a b : Additive X) : edist a.toMul b.toMul = edist a b :=
  rfl


@[simp]
theorem edist_toAdd (a b : Multiplicative X) : edist a.toAdd b.toAdd = edist a b :=
  rfl


instance [PseudoEMetricSpace X] : PseudoEMetricSpace (Additive X) := ‹PseudoEMetricSpace X›

instance [PseudoEMetricSpace X] : PseudoEMetricSpace (Multiplicative X) := ‹PseudoEMetricSpace X›

instance [EMetricSpace X] : EMetricSpace (Additive X) := ‹EMetricSpace X›

instance [EMetricSpace X] : EMetricSpace (Multiplicative X) := ‹EMetricSpace X›


instance : EDist Xᵒᵈ := ‹EDist X›


@[simp]
theorem edist_toDual (a b : X) : edist (toDual a) (toDual b) = edist a b :=
  rfl


@[simp]
theorem edist_ofDual (a b : Xᵒᵈ) : edist (ofDual a) (ofDual b) = edist a b :=
  rfl


instance [PseudoEMetricSpace X] : PseudoEMetricSpace Xᵒᵈ := ‹PseudoEMetricSpace X›

instance [EMetricSpace X] : EMetricSpace Xᵒᵈ := ‹EMetricSpace X›

