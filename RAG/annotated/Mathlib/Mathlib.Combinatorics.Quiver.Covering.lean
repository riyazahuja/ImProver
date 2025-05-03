/-- The `Quiver.Star` at a vertex is the collection of arrows whose source is the vertex.
The type `Quiver.Star u` is defined to be `Σ (v : U), (u ⟶ v)`. -/
abbrev Quiver.Star (u : U) :=
  Σ v : U, u ⟶ v


/-- Constructor for `Quiver.Star`. Defined to be `Sigma.mk`. -/
protected abbrev Quiver.Star.mk {u v : U} (f : u ⟶ v) : Quiver.Star u :=
  ⟨_, f⟩


/-- The `Quiver.Costar` at a vertex is the collection of arrows whose target is the vertex.
The type `Quiver.Costar v` is defined to be `Σ (u : U), (u ⟶ v)`. -/
abbrev Quiver.Costar (v : U) :=
  Σ u : U, u ⟶ v


/-- Constructor for `Quiver.Costar`. Defined to be `Sigma.mk`. -/
protected abbrev Quiver.Costar.mk {u v : U} (f : u ⟶ v) : Quiver.Costar v :=
  ⟨_, f⟩


/-- A prefunctor induces a map of `Quiver.Star` at every vertex. -/
@[simps]
def Prefunctor.star (u : U) : Quiver.Star u → Quiver.Star (φ.obj u) := fun F =>
  Quiver.Star.mk (φ.map F.2)


/-- A prefunctor induces a map of `Quiver.Costar` at every vertex. -/
@[simps]
def Prefunctor.costar (u : U) : Quiver.Costar u → Quiver.Costar (φ.obj u) := fun F =>
  Quiver.Costar.mk (φ.map F.2)


@[simp]
theorem Prefunctor.star_apply {u v : U} (e : u ⟶ v) :
    φ.star u (Quiver.Star.mk e) = Quiver.Star.mk (φ.map e) :=
  rfl


@[simp]
theorem Prefunctor.costar_apply {u v : U} (e : u ⟶ v) :
    φ.costar v (Quiver.Costar.mk e) = Quiver.Costar.mk (φ.map e) :=
  rfl


theorem Prefunctor.star_comp (u : U) : (φ ⋙q ψ).star u = ψ.star (φ.obj u) ∘ φ.star u :=
  rfl


theorem Prefunctor.costar_comp (u : U) : (φ ⋙q ψ).costar u = ψ.costar (φ.obj u) ∘ φ.costar u :=
  rfl


/-- A prefunctor is a covering of quivers if it defines bijections on all stars and costars. -/
protected structure Prefunctor.IsCovering : Prop where
  star_bijective : ∀ u, Bijective (φ.star u)
  costar_bijective : ∀ u, Bijective (φ.costar u)


@[simp]
theorem Prefunctor.IsCovering.map_injective (hφ : φ.IsCovering) {u v : U} :
    Injective fun f : u ⟶ v => φ.map f := by
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : φ.IsCovering
    u v : U
    ⊢ Function.Injective fun f => φ.map f
  -/
  rintro f g he
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : φ.IsCovering
    u v : U
    f g : Quiver.Hom u v
    he : Eq ((fun f => φ.map f) f) ((fun f => φ.map f) g)
    ⊢ Eq f g
  -/
  have : φ.star u (Quiver.Star.mk f) = φ.star u (Quiver.Star.mk g) := by simpa using he
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : φ.IsCovering
    u v : U
    f g : Quiver.Hom u v
    he : Eq ((fun f => φ.map f) f) ((fun f => φ.map f) g)
    this : Eq (φ.star u (Quiver.Star.mk f)) (φ.star u (Quiver.Star.mk g))
    ⊢ Eq f g
  -/
  simpa using (hφ.star_bijective u).left this
  /-
    🎉 no goals
  -/


theorem Prefunctor.IsCovering.comp (hφ : φ.IsCovering) (hψ : ψ.IsCovering) : (φ ⋙q ψ).IsCovering :=
  ⟨fun _ => (hψ.star_bijective _).comp (hφ.star_bijective _),
   fun _ => (hψ.costar_bijective _).comp (hφ.costar_bijective _)⟩


theorem Prefunctor.IsCovering.of_comp_right (hψ : ψ.IsCovering) (hφψ : (φ ⋙q ψ).IsCovering) :
    φ.IsCovering :=
  ⟨fun _ => (Bijective.of_comp_iff' (hψ.star_bijective _) _).mp (hφψ.star_bijective _),
   fun _ => (Bijective.of_comp_iff' (hψ.costar_bijective _) _).mp (hφψ.costar_bijective _)⟩


theorem Prefunctor.IsCovering.of_comp_left (hφ : φ.IsCovering) (hφψ : (φ ⋙q ψ).IsCovering)
    (φsur : Surjective φ.obj) : ψ.IsCovering := by
  /-
    U : Type u_1
    inst✝² : Quiver U
    V : Type u_2
    inst✝¹ : Quiver V
    φ : Prefunctor U V
    W : Type u_3
    inst✝ : Quiver W
    ψ : Prefunctor V W
    hφ : φ.IsCovering
    hφψ : (φ.comp ψ).IsCovering
    φsur : Function.Surjective φ.obj
    ⊢ ψ.IsCovering
  -/
  refine ⟨fun v => ?_, fun v => ?_⟩ <;> obtain ⟨u, rfl⟩ := φsur v
  exacts [(Bijective.of_comp_iff _ (hφ.star_bijective u)).mp (hφψ.star_bijective u),
    (Bijective.of_comp_iff _ (hφ.costar_bijective u)).mp (hφψ.costar_bijective u)]


/-- The star of the symmetrification of a quiver at a vertex `u` is equivalent to the sum of the
star and the costar at `u` in the original quiver. -/
def Quiver.symmetrifyStar (u : U) :
    Quiver.Star (Symmetrify.of.obj u) ≃ Quiver.Star u ⊕ Quiver.Costar u :=
  Equiv.sigmaSumDistrib _ _


/-- The costar of the symmetrification of a quiver at a vertex `u` is equivalent to the sum of the
costar and the star at `u` in the original quiver. -/
def Quiver.symmetrifyCostar (u : U) :
    Quiver.Costar (Symmetrify.of.obj u) ≃ Quiver.Costar u ⊕ Quiver.Star u :=
  Equiv.sigmaSumDistrib _ _


theorem Prefunctor.symmetrifyStar (u : U) :
    φ.symmetrify.star u =
      (Quiver.symmetrifyStar _).symm ∘ Sum.map (φ.star u) (φ.costar u) ∘
        Quiver.symmetrifyStar u := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    u : U
    ⊢ Eq (φ.symmetrify.star u) (Function.comp (⇑(Quiver.symmetrifyStar (φ.obj u)). …
  -/
  erw [Equiv.eq_symm_comp]
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    u : U
    ⊢ Eq (Function.comp (⇑(Quiver.symmetrifyStar (φ.obj u))) (φ.symmetrify.star u) …
  -/
  ext ⟨v, f | g⟩ <;>
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [Quiver.symmetrifyStar]`
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom u v
      ⊢ Eq (Function.comp (⇑(Quiver.symmetrifyStar (φ.obj u))) (φ.symmetrify.star u) …
    -/
    simp only [Quiver.symmetrifyStar, Function.comp_apply] <;>
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom u v
      ⊢ Eq ((Equiv.sigmaSumDistrib (Quiver.Hom (Quiver.Symmetrify.of.obj (φ.obj u))) …
    -/
    erw [Equiv.sigmaSumDistrib_apply, Equiv.sigmaSumDistrib_apply] <;>
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom u v
      ⊢ Eq (Sum.map (Sigma.mk (φ.symmetrify.star u ⟨v, Sum.inl f⟩).fst) (Sigma.mk (φ …
    -/
    /-
      🎉 no goals
    -/
    simp
    /-
      🎉 no goals
    -/


protected theorem Prefunctor.symmetrifyCostar (u : U) :
    φ.symmetrify.costar u =
      (Quiver.symmetrifyCostar _).symm ∘
        Sum.map (φ.costar u) (φ.star u) ∘ Quiver.symmetrifyCostar u := by
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    u : U
    ⊢ Eq (φ.symmetrify.costar u) (Function.comp (⇑(Quiver.symmetrifyCostar (φ.obj  …
  -/
  erw [Equiv.eq_symm_comp]
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    u : U
    ⊢ Eq (Function.comp (⇑(Quiver.symmetrifyCostar (φ.obj u))) (φ.symmetrify.costa …
  -/
  ext ⟨v, f | g⟩ <;>
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [Quiver.symmetrifyCostar]`
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom v u
      ⊢ Eq (Function.comp (⇑(Quiver.symmetrifyCostar (φ.obj u))) (φ.symmetrify.costa …
    -/
    simp only [Quiver.symmetrifyCostar, Function.comp_apply] <;>
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom v u
      ⊢ Eq ((Equiv.sigmaSumDistrib (fun u_1 => Quiver.Hom u_1 (Quiver.Symmetrify.of. …
    -/
    erw [Equiv.sigmaSumDistrib_apply, Equiv.sigmaSumDistrib_apply] <;>
    /-
      case h.mk.inl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      u : U
      v : Quiver.Symmetrify U
      f : Quiver.Hom v u
      ⊢ Eq (Sum.map (Sigma.mk (φ.symmetrify.costar u ⟨v, Sum.inl f⟩).fst) (Sigma.mk  …
    -/
    /-
      🎉 no goals
    -/
    simp
    /-
      🎉 no goals
    -/


protected theorem Prefunctor.IsCovering.symmetrify (hφ : φ.IsCovering) :
    φ.symmetrify.IsCovering := by
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : φ.IsCovering
    ⊢ φ.symmetrify.IsCovering
  -/
  refine ⟨fun u => ?_, fun u => ?_⟩ <;>
    -- Porting note: was
    -- simp [φ.symmetrifyStar, φ.symmetrifyCostar, hφ.star_bijective u, hφ.costar_bijective u]
    /-
      case refine_1
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : φ.IsCovering
      u : Quiver.Symmetrify U
      ⊢ Function.Bijective (φ.symmetrify.star u)
    -/
    simp only [φ.symmetrifyStar, φ.symmetrifyCostar] <;>
    /-
      case refine_1
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : φ.IsCovering
      u : Quiver.Symmetrify U
      ⊢ Function.Bijective (Function.comp (⇑(Quiver.symmetrifyStar (φ.obj u)).symm)  …
    -/
    erw [EquivLike.comp_bijective, EquivLike.bijective_comp] <;>
    /-
      case refine_1
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : φ.IsCovering
      u : Quiver.Symmetrify U
      ⊢ Function.Bijective (Sum.map (φ.star u) (φ.costar u))
    -/
    /-
      🎉 no goals
    -/
    simp [hφ.star_bijective u, hφ.costar_bijective u]
    /-
      🎉 no goals
    -/


/-- The path star at a vertex `u` is the type of all paths starting at `u`.
The type `Quiver.PathStar u` is defined to be `Σ v : U, Path u v`. -/
abbrev Quiver.PathStar (u : U) :=
  Σ v : U, Path u v


/-- Constructor for `Quiver.PathStar`. Defined to be `Sigma.mk`. -/
protected abbrev Quiver.PathStar.mk {u v : U} (p : Path u v) : Quiver.PathStar u :=
  ⟨_, p⟩


/-- A prefunctor induces a map of path stars. -/
def Prefunctor.pathStar (u : U) : Quiver.PathStar u → Quiver.PathStar (φ.obj u) := fun p =>
  Quiver.PathStar.mk (φ.mapPath p.2)


@[simp]
theorem Prefunctor.pathStar_apply {u v : U} (p : Path u v) :
    φ.pathStar u (Quiver.PathStar.mk p) = Quiver.PathStar.mk (φ.mapPath p) :=
  rfl


theorem Prefunctor.pathStar_injective (hφ : ∀ u, Injective (φ.star u)) (u : U) :
    Injective (φ.pathStar u) := by
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Injective (φ.star u)
    u : U
    ⊢ Function.Injective (φ.pathStar u)
  -/
  dsimp (config := { unfoldPartialApp := true }) [Prefunctor.pathStar, Quiver.PathStar.mk]
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Injective (φ.star u)
    u : U
    ⊢ Function.Injective fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩
  -/
  rintro ⟨v₁, p₁⟩
  /-
    case mk
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Injective (φ.star u)
    u v₁ : U
    p₁ : Quiver.Path u v₁
    ⊢ ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨v …
  -/
  induction' p₁ with x₁ y₁ p₁ e₁ ih <;>
    /-
      case mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ : U
      ⊢ ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u …
    -/
    rintro ⟨y₂, p₂⟩ <;>
    /-
      case mk.nil.mk
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ y₂ : U
      p₂ : Quiver.Path u y₂
      ⊢ Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u, Quiver.Path.nil⟩) ((fun p  …
    -/
    cases' p₂ with x₂ _ p₂ e₂ <;>
    /-
      case mk.nil.mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ : U
      ⊢ Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u, Quiver.Path.nil⟩) ((fun p  …
    -/
    intro h <;>
    -- Porting note: added `Sigma.mk.inj_iff`
    simp only [Prefunctor.pathStar_apply, Prefunctor.mapPath_nil, Prefunctor.mapPath_cons,
      Sigma.mk.inj_iff] at h
  · -- Porting note: goal not present in lean3.
    /-
      case mk.nil.mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ : U
      h : True
      ⊢ Eq ⟨u, Quiver.Path.nil⟩ ⟨u, Quiver.Path.nil⟩
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.nil.mk.cons
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      h : And (Eq (φ.obj u) (φ.obj y₂)) (HEq Quiver.Path.nil ((φ.mapPath p₂).cons (φ …
      ⊢ Eq ⟨u, Quiver.Path.nil⟩ ⟨y₂, p₂.cons e₂⟩
    -/
  · exfalso
    /-
      case mk.nil.mk.cons
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      h : And (Eq (φ.obj u) (φ.obj y₂)) (HEq Quiver.Path.nil ((φ.mapPath p₂).cons (φ …
      ⊢ False
    -/
    cases' h with h h'
    /-
      case mk.nil.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      h : Eq (φ.obj u) (φ.obj y₂)
      h' : HEq Quiver.Path.nil ((φ.mapPath p₂).cons (φ.map e₂))
      ⊢ False
    -/
    rw [← Path.eq_cast_iff_heq rfl h.symm, Path.cast_cons] at h'
    /-
      case mk.nil.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      h : Eq (φ.obj u) (φ.obj y₂)
      h' : Eq Quiver.Path.nil ((Quiver.Path.cast ⋯ ⋯ (φ.mapPath p₂)).cons (Quiver.Ho …
      ⊢ False
    -/
    exact (Path.nil_ne_cons _ _) h'
    /-
      🎉 no goals
    -/
    /-
      case mk.cons.mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      h : And (Eq (φ.obj y₁) (φ.obj u)) (HEq ((φ.mapPath p₁).cons (φ.map e₁)) Quiver …
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨u, Quiver.Path.nil⟩
    -/
  · exfalso
    /-
      case mk.cons.mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      h : And (Eq (φ.obj y₁) (φ.obj u)) (HEq ((φ.mapPath p₁).cons (φ.map e₁)) Quiver …
      ⊢ False
    -/
    cases' h with h h'
    /-
      case mk.cons.mk.nil.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      h : Eq (φ.obj y₁) (φ.obj u)
      h' : HEq ((φ.mapPath p₁).cons (φ.map e₁)) Quiver.Path.nil
      ⊢ False
    -/
    rw [← Path.cast_eq_iff_heq rfl h, Path.cast_cons] at h'
    /-
      case mk.cons.mk.nil.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      h : Eq (φ.obj y₁) (φ.obj u)
      h' : Eq ((Quiver.Path.cast ⋯ ⋯ (φ.mapPath p₁)).cons (Quiver.Hom.cast ⋯ h (φ.ma …
      ⊢ False
    -/
    exact (Path.cons_ne_nil _ _) h'
    /-
      🎉 no goals
    -/
    /-
      case mk.cons.mk.cons
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      h : And (Eq (φ.obj y₁) (φ.obj y₂)) (HEq ((φ.mapPath p₁).cons (φ.map e₁)) ((φ.m …
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
  · cases' h with hφy h'
    /-
      case mk.cons.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      h' : HEq ((φ.mapPath p₁).cons (φ.map e₁)) ((φ.mapPath p₂).cons (φ.map e₂))
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
    rw [← Path.cast_eq_iff_heq rfl hφy, Path.cast_cons, Path.cast_rfl_rfl] at h'
    /-
      case mk.cons.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
    have hφx := Path.obj_eq_of_cons_eq_cons h'
    /-
      case mk.cons.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      hφx : Eq (φ.obj x₁) (φ.obj x₂)
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
    have hφp := Path.heq_of_cons_eq_cons h'
    /-
      case mk.cons.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      hφx : Eq (φ.obj x₁) (φ.obj x₂)
      hφp : HEq (φ.mapPath p₁) (φ.mapPath p₂)
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
    have hφe := HEq.trans (Hom.cast_heq rfl hφy _).symm (Path.hom_heq_of_cons_eq_cons h')
    have h_path_star : φ.pathStar u ⟨x₁, p₁⟩ = φ.pathStar u ⟨x₂, p₂⟩ := by
      simp only [Prefunctor.pathStar_apply, Sigma.mk.inj_iff]; exact ⟨hφx, hφp⟩
    /-
      case mk.cons.mk.cons.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ x₂ : U
      p₂ : Quiver.Path u x₂
      e₂ : Quiver.Hom x₂ y₂
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      hφx : Eq (φ.obj x₁) (φ.obj x₂)
      hφp : HEq (φ.mapPath p₁) (φ.mapPath p₂)
      hφe : HEq (φ.map e₁) (φ.map e₂)
      h_path_star : Eq (φ.pathStar u ⟨x₁, p₁⟩) (φ.pathStar u ⟨x₂, p₂⟩)
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₂.cons e₂⟩
    -/
    cases ih h_path_star
    have h_star : φ.star x₁ ⟨y₁, e₁⟩ = φ.star x₁ ⟨y₂, e₂⟩ := by
      simp only [Prefunctor.star_apply, Sigma.mk.inj_iff]; exact ⟨hφy, hφe⟩
    /-
      case mk.cons.mk.cons.intro.refl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      y₂ : U
      hφy : Eq (φ.obj y₁) (φ.obj y₂)
      e₂ : Quiver.Hom x₁ y₂
      hφx : Eq (φ.obj x₁) (φ.obj x₁)
      hφe : HEq (φ.map e₁) (φ.map e₂)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      hφp : HEq (φ.mapPath p₁) (φ.mapPath p₁)
      h_path_star : Eq (φ.pathStar u ⟨x₁, p₁⟩) (φ.pathStar u ⟨x₁, p₁⟩)
      h_star : Eq (φ.star x₁ ⟨y₁, e₁⟩) (φ.star x₁ ⟨y₂, e₂⟩)
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₂, p₁.cons e₂⟩
    -/
    cases hφ x₁ h_star
    /-
      case mk.cons.mk.cons.intro.refl.refl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Injective (φ.star u)
      u v₁ x₁ y₁ : U
      p₁ : Quiver.Path u x₁
      e₁ : Quiver.Hom x₁ y₁
      ih : ∀ ⦃a₂ : Quiver.PathStar u⦄, Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) …
      hφx : Eq (φ.obj x₁) (φ.obj x₁)
      hφp : HEq (φ.mapPath p₁) (φ.mapPath p₁)
      h_path_star : Eq (φ.pathStar u ⟨x₁, p₁⟩) (φ.pathStar u ⟨x₁, p₁⟩)
      hφy : Eq (φ.obj y₁) (φ.obj y₁)
      hφe : HEq (φ.map e₁) (φ.map e₁)
      h' : Eq ((φ.mapPath p₁).cons (Quiver.Hom.cast ⋯ hφy (φ.map e₁))) ((φ.mapPath p …
      h_star : Eq (φ.star x₁ ⟨y₁, e₁⟩) (φ.star x₁ ⟨y₁, e₁⟩)
      ⊢ Eq ⟨y₁, p₁.cons e₁⟩ ⟨y₁, p₁.cons e₁⟩
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Prefunctor.pathStar_surjective (hφ : ∀ u, Surjective (φ.star u)) (u : U) :
    Surjective (φ.pathStar u) := by
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Surjective (φ.star u)
    u : U
    ⊢ Function.Surjective (φ.pathStar u)
  -/
  dsimp (config := { unfoldPartialApp := true }) [Prefunctor.pathStar, Quiver.PathStar.mk]
  /-
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Surjective (φ.star u)
    u : U
    ⊢ Function.Surjective fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩
  -/
  rintro ⟨v, p⟩
  /-
    case mk
    U : Type u_1
    inst✝¹ : Quiver U
    V : Type u_2
    inst✝ : Quiver V
    φ : Prefunctor U V
    hφ : ∀ (u : U), Function.Surjective (φ.star u)
    u : U
    v : V
    p : Quiver.Path (φ.obj u) v
    ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v, p⟩
  -/
  induction' p with v' v'' p' ev ih
    /-
      case mk.nil
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨φ.obj u, Q …
    -/
  · use ⟨u, Path.nil⟩
    /-
      case h
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      ⊢ Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u, Quiver.Path.nil⟩) ⟨φ.obj u …
    -/
    simp only [Prefunctor.mapPath_nil, eq_self_iff_true, heq_iff_eq, and_self_iff]
    /-
      🎉 no goals
    -/
    /-
      case mk.cons
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v' v'' : V
      p' : Quiver.Path (φ.obj u) v'
      ev : Quiver.Hom v' v''
      ih : Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v', p'⟩
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', p'.co …
    -/
  · obtain ⟨⟨u', q'⟩, h⟩ := ih
    /-
      case mk.cons.intro.mk
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v' v'' : V
      p' : Quiver.Path (φ.obj u) v'
      ev : Quiver.Hom v' v''
      u' : U
      q' : Quiver.Path u u'
      h : Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u', q'⟩) ⟨v', p'⟩
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', p'.co …
    -/
    simp only at h
    /-
      case mk.cons.intro.mk
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v' v'' : V
      p' : Quiver.Path (φ.obj u) v'
      ev : Quiver.Hom v' v''
      u' : U
      q' : Quiver.Path u u'
      h : Eq ⟨φ.obj u', φ.mapPath q'⟩ ⟨v', p'⟩
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', p'.co …
    -/
    obtain ⟨rfl, rfl⟩ := h
    /-
      case mk.cons.intro.mk.refl
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v'' : V
      u' : U
      q' : Quiver.Path u u'
      ev : Quiver.Hom (φ.obj u') v''
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', (φ.ma …
    -/
    obtain ⟨⟨u'', eu⟩, k⟩ := hφ u' ⟨_, ev⟩
    /-
      case mk.cons.intro.mk.refl.intro.mk
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v'' : V
      u' : U
      q' : Quiver.Path u u'
      ev : Quiver.Hom (φ.obj u') v''
      u'' : U
      eu : Quiver.Hom u' u''
      k : Eq (φ.star u' ⟨u'', eu⟩) ⟨v'', ev⟩
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', (φ.ma …
    -/
    simp only [star_apply, Sigma.mk.inj_iff] at k
    -- Porting note: was `obtain ⟨rfl, rfl⟩ := k`
    /-
      case mk.cons.intro.mk.refl.intro.mk
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v v'' : V
      u' : U
      q' : Quiver.Path u u'
      ev : Quiver.Hom (φ.obj u') v''
      u'' : U
      eu : Quiver.Hom u' u''
      k : And (Eq (φ.obj u'') v'') (HEq (φ.map eu) ev)
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨v'', (φ.ma …
    -/
    obtain ⟨rfl, k⟩ := k
    /-
      case mk.cons.intro.mk.refl.intro.mk.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      u' : U
      q' : Quiver.Path u u'
      u'' : U
      eu : Quiver.Hom u' u''
      ev : Quiver.Hom (φ.obj u') (φ.obj u'')
      k : HEq (φ.map eu) ev
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨φ.obj u'', …
    -/
    simp only [heq_eq_eq] at k
    /-
      case mk.cons.intro.mk.refl.intro.mk.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      u' : U
      q' : Quiver.Path u u'
      u'' : U
      eu : Quiver.Hom u' u''
      ev : Quiver.Hom (φ.obj u') (φ.obj u'')
      k : Eq (φ.map eu) ev
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨φ.obj u'', …
    -/
    subst k
    /-
      case mk.cons.intro.mk.refl.intro.mk.intro
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      u' : U
      q' : Quiver.Path u u'
      u'' : U
      eu : Quiver.Hom u' u''
      ⊢ Exists fun a => Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) a) ⟨φ.obj u'', …
    -/
    use ⟨_, q'.cons eu⟩
    /-
      case h
      U : Type u_1
      inst✝¹ : Quiver U
      V : Type u_2
      inst✝ : Quiver V
      φ : Prefunctor U V
      hφ : ∀ (u : U), Function.Surjective (φ.star u)
      u : U
      v : V
      u' : U
      q' : Quiver.Path u u'
      u'' : U
      eu : Quiver.Hom u' u''
      ⊢ Eq ((fun p => ⟨φ.obj p.fst, φ.mapPath p.snd⟩) ⟨u'', q'.cons eu⟩) ⟨φ.obj u'', …
    -/
    simp only [Prefunctor.mapPath_cons, eq_self_iff_true, heq_iff_eq, and_self_iff]
    /-
      🎉 no goals
    -/


theorem Prefunctor.pathStar_bijective (hφ : ∀ u, Bijective (φ.star u)) (u : U) :
    Bijective (φ.pathStar u) :=
  ⟨φ.pathStar_injective (fun u => (hφ u).1) _, φ.pathStar_surjective (fun u => (hφ u).2) _⟩


protected theorem pathStar_bijective (hφ : φ.IsCovering) (u : U) : Bijective (φ.pathStar u) :=
  φ.pathStar_bijective hφ.1 u


/-- In a quiver with involutive inverses, the star and costar at every vertex are equivalent.
This map is induced by `Quiver.reverse`. -/
@[simps]
def Quiver.starEquivCostar (u : U) : Quiver.Star u ≃ Quiver.Costar u where
  toFun e := ⟨e.1, reverse e.2⟩
  invFun e := ⟨e.1, reverse e.2⟩
                   /-
                     U : Type ?u.17438
                     inst✝⁴ : Quiver U
                     V : Type ?u.17444
                     inst✝³ : Quiver V
                     φ : Prefunctor U V
                     W : Type ?u.17468
                     inst✝² : Quiver W
                     ψ : Prefunctor V W
                     inst✝¹ : Quiver.HasInvolutiveReverse U
                     inst✝ : Quiver.HasInvolutiveReverse V
                     u : U
                     e : Quiver.Star u
                     ⊢ Eq ((fun e => ⟨e.fst, Quiver.reverse e.snd⟩) ((fun e => ⟨e.fst, Quiver.rever …
                   -/
  left_inv e := by simp [Sigma.ext_iff]
                   /-
                     🎉 no goals
                   -/
                    /-
                      U : Type ?u.17438
                      inst✝⁴ : Quiver U
                      V : Type ?u.17444
                      inst✝³ : Quiver V
                      φ : Prefunctor U V
                      W : Type ?u.17468
                      inst✝² : Quiver W
                      ψ : Prefunctor V W
                      inst✝¹ : Quiver.HasInvolutiveReverse U
                      inst✝ : Quiver.HasInvolutiveReverse V
                      u : U
                      e : Quiver.Costar u
                      ⊢ Eq ((fun e => ⟨e.fst, Quiver.reverse e.snd⟩) ((fun e => ⟨e.fst, Quiver.rever …
                    -/
  right_inv e := by simp [Sigma.ext_iff]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem Quiver.starEquivCostar_apply {u v : U} (e : u ⟶ v) :
    Quiver.starEquivCostar u (Quiver.Star.mk e) = Quiver.Costar.mk (reverse e) :=
  rfl


@[simp]
theorem Quiver.starEquivCostar_symm_apply {u v : U} (e : u ⟶ v) :
    (Quiver.starEquivCostar v).symm (Quiver.Costar.mk e) = Quiver.Star.mk (reverse e) :=
  rfl


theorem Prefunctor.costar_conj_star (u : U) :
    φ.costar u = Quiver.starEquivCostar (φ.obj u) ∘ φ.star u ∘ (Quiver.starEquivCostar u).symm := by
  /-
    U : Type u_1
    inst✝⁴ : Quiver U
    V : Type u_2
    inst✝³ : Quiver V
    φ : Prefunctor U V
    inst✝² : Quiver.HasInvolutiveReverse U
    inst✝¹ : Quiver.HasInvolutiveReverse V
    inst✝ : φ.MapReverse
    u : U
    ⊢ Eq (φ.costar u) (Function.comp (⇑(Quiver.starEquivCostar (φ.obj u))) (Functi …
  -/
                 /-
                   🎉 no goals
                 -/
  ext ⟨v, f⟩ <;> simp
                 /-
                   🎉 no goals
                 -/


theorem Prefunctor.bijective_costar_iff_bijective_star (u : U) :
    Bijective (φ.costar u) ↔ Bijective (φ.star u) := by
  /-
    U : Type u_1
    inst✝⁴ : Quiver U
    V : Type u_2
    inst✝³ : Quiver V
    φ : Prefunctor U V
    inst✝² : Quiver.HasInvolutiveReverse U
    inst✝¹ : Quiver.HasInvolutiveReverse V
    inst✝ : φ.MapReverse
    u : U
    ⊢ Iff (Function.Bijective (φ.costar u)) (Function.Bijective (φ.star u))
  -/
  rw [Prefunctor.costar_conj_star φ, EquivLike.comp_bijective, EquivLike.bijective_comp]
  /-
    🎉 no goals
  -/


theorem Prefunctor.isCovering_of_bijective_star (h : ∀ u, Bijective (φ.star u)) : φ.IsCovering :=
  ⟨h, fun u => (φ.bijective_costar_iff_bijective_star u).2 (h u)⟩


theorem Prefunctor.isCovering_of_bijective_costar (h : ∀ u, Bijective (φ.costar u)) :
    φ.IsCovering :=
  ⟨fun u => (φ.bijective_costar_iff_bijective_star u).1 (h u), h⟩


