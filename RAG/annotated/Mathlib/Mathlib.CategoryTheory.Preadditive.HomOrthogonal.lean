/-- A family of objects is "hom orthogonal" if
there is at most one morphism between distinct objects.

(In a category with zero morphisms, that must be the zero morphism.) -/
def HomOrthogonal {ι : Type*} (s : ι → C) : Prop :=
  Pairwise fun i j => Subsingleton (s i ⟶ s j)


theorem eq_zero [HasZeroMorphisms C] (o : HomOrthogonal s) {i j : ι} (w : i ≠ j) (f : s i ⟶ s j) :
    f = 0 :=
  (o w).elim _ _


open scoped Classical in
/-- Morphisms between two direct sums over a hom orthogonal family `s : ι → C`
are equivalent to block diagonal matrices,
with blocks indexed by `ι`,
and matrix entries in `i`-th block living in the endomorphisms of `s i`. -/
@[simps]
noncomputable def matrixDecomposition (o : HomOrthogonal s) {α β : Type} [Finite α] [Finite β]
    {f : α → ι} {g : β → ι} :
    ((⨁ fun a => s (f a)) ⟶ ⨁ fun b => s (g b)) ≃
      ∀ i : ι, Matrix (g ⁻¹' {i}) (f ⁻¹' {i}) (End (s i)) where
  toFun z i j k :=
    eqToHom
        (by
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            ι : Type u_1
            s : ι → C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
            o : CategoryTheory.HomOrthogonal s
            α β : Type
            inst✝¹ : Finite α
            inst✝ : Finite β
            f : α → ι
            g : β → ι
            z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
            i : ι
            j : ↑(Set.preimage g (Singleton.singleton i))
            k : ↑(Set.preimage f (Singleton.singleton i))
            ⊢ Eq (s i) (s (f ↑k))
          -/
          rcases k with ⟨k, ⟨⟩⟩
          /-
            case mk.refl
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            ι : Type u_1
            s : ι → C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
            o : CategoryTheory.HomOrthogonal s
            α β : Type
            inst✝¹ : Finite α
            inst✝ : Finite β
            f : α → ι
            g : β → ι
            z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
            k : α
            j : ↑(Set.preimage g (Singleton.singleton (f k)))
            ⊢ Eq (s (f k)) (s (f ↑⟨k, ⋯⟩))
          -/
          simp) ≫
          /-
            🎉 no goals
          -/
      biproduct.components z k j ≫
        eqToHom
          (by
            /-
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              ι : Type u_1
              s : ι → C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
              o : CategoryTheory.HomOrthogonal s
              α β : Type
              inst✝¹ : Finite α
              inst✝ : Finite β
              f : α → ι
              g : β → ι
              z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
              i : ι
              j : ↑(Set.preimage g (Singleton.singleton i))
              k : ↑(Set.preimage f (Singleton.singleton i))
              ⊢ Eq (s (g ↑j)) (s i)
            -/
            rcases j with ⟨j, ⟨⟩⟩
            /-
              case mk.refl
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              ι : Type u_1
              s : ι → C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
              o : CategoryTheory.HomOrthogonal s
              α β : Type
              inst✝¹ : Finite α
              inst✝ : Finite β
              f : α → ι
              g : β → ι
              z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
              j : β
              k : ↑(Set.preimage f (Singleton.singleton (g j)))
              ⊢ Eq (s (g ↑⟨j, ⋯⟩)) (s (g j))
            -/
            simp)
            /-
              🎉 no goals
            -/
  invFun z :=
    biproduct.matrix fun j k =>
                                           /-
                                             C : Type u
                                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                                             ι : Type u_1
                                             s : ι → C
                                             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                             inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
                                             o : CategoryTheory.HomOrthogonal s
                                             α β : Type
                                             inst✝¹ : Finite α
                                             inst✝ : Finite β
                                             f : α → ι
                                             g : β → ι
                                             z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
                                             j : α
                                             k : β
                                             h : Eq (f j) (g k)
                                             ⊢ Membership.mem (Set.preimage g (Singleton.singleton (f j))) k
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                                            /-
                                                              🎉 no goals
                                                            -/
      if h : f j = g k then z (f j) ⟨k, by simp [h]⟩ ⟨j, by simp⟩ ≫ eqToHom (by simp [h]) else 0
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  left_inv z := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      ⊢ Eq ((fun z => CategoryTheory.Limits.biproduct.matrix fun j k => dite (Eq (f  …
    -/
    ext j k
    /-
      case w.w
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      j : β
      k : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
    -/
    simp only [biproduct.matrix_π, biproduct.ι_desc]
    /-
      case w.w
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      j : β
      k : α
      ⊢ Eq (dite (Eq (f k) (g j)) (fun h => CategoryTheory.CategoryStruct.comp (Cate …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        j : β
        k : α
        h : Eq (f k) (g j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        case pos
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        j : β
        k : α
        h : Eq (f k) (g j)
        ⊢ Eq (CategoryTheory.Limits.biproduct.components z k j) (CategoryTheory.Catego …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        j : β
        k : α
        h : Not (Eq (f k) (g j))
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι  …
      -/
    · symm
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        j : β
        k : α
        h : Not (Eq (f k) (g j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
      -/
      apply o.eq_zero h
      /-
        🎉 no goals
      -/
  right_inv z := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
      ⊢ Eq ((fun z i j k => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToH …
    -/
    ext i ⟨j, w⟩ ⟨k, ⟨⟩⟩
    /-
      case h.h.mk.h.mk.refl
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
      j : β
      k : α
      w : Membership.mem (Set.preimage g (Singleton.singleton (f k))) j
      ⊢ Eq ((fun z i j k => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToH …
    -/
    simp only [eqToHom_refl, biproduct.matrix_components, Category.id_comp]
    /-
      case h.h.mk.h.mk.refl
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β : Type
      inst✝¹ : Finite α
      inst✝ : Finite β
      f : α → ι
      g : β → ι
      z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
      j : β
      k : α
      w : Membership.mem (Set.preimage g (Singleton.singleton (f k))) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq (f k) (g j)) (fun h => Cate …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
        j : β
        k : α
        w : Membership.mem (Set.preimage g (Singleton.singleton (f k))) j
        h : Eq (f k) (g j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
        j : β
        k : α
        w : Membership.mem (Set.preimage g (Singleton.singleton (f k))) j
        h : Not (Eq (f k) (g j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.eqToHom ⋯)) (z (f k …
      -/
    · exfalso
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        z : (i : ι) → Matrix (↑(Set.preimage g (Singleton.singleton i))) (↑(Set.preima …
        j : β
        k : α
        w : Membership.mem (Set.preimage g (Singleton.singleton (f k))) j
        h : Not (Eq (f k) (g j))
        ⊢ False
      -/
      exact h w.symm
      /-
        🎉 no goals
      -/


/-- `HomOrthogonal.matrixDecomposition` as an additive equivalence. -/
@[simps!]
noncomputable def matrixDecompositionAddEquiv (o : HomOrthogonal s) {α β : Type} [Finite α]
    [Finite β] {f : α → ι} {g : β → ι} :
    ((⨁ fun a => s (f a)) ⟶ ⨁ fun b => s (g b)) ≃+
      ∀ i : ι, Matrix (g ⁻¹' {i}) (f ⁻¹' {i}) (End (s i)) :=
  { o.matrixDecomposition with
    map_add' := fun w z => by
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryT …
        ⊢ Eq (__src✝.toFun (HAdd.hAdd w z)) (HAdd.hAdd (__src✝.toFun w) (__src✝.toFun  …
      -/
      ext
      /-
        case h.a
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryT …
        x✝ : ι
        i✝ : ↑(Set.preimage g (Singleton.singleton x✝))
        j✝ : ↑(Set.preimage f (Singleton.singleton x✝))
        ⊢ Eq (__src✝.toFun (HAdd.hAdd w z) x✝ i✝ j✝) (HAdd.hAdd (__src✝.toFun w) (__sr …
      -/
      dsimp [biproduct.components]
      /-
        case h.a
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryT …
        x✝ : ι
        i✝ : ↑(Set.preimage g (Singleton.singleton x✝))
        j✝ : ↑(Set.preimage f (Singleton.singleton x✝))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp }
      /-
        🎉 no goals
      -/


open scoped Classical in
@[simp]
theorem matrixDecomposition_id (o : HomOrthogonal s) {α : Type} [Finite α] {f : α → ι} (i : ι) :
    o.matrixDecomposition (𝟙 (⨁ fun a => s (f a))) i = 1 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α : Type
    inst✝ : Finite α
    f : α → ι
    i : ι
    ⊢ Eq (o.matrixDecomposition (CategoryTheory.CategoryStruct.id (CategoryTheory. …
  -/
  ext ⟨b, ⟨⟩⟩ ⟨a, j_property⟩
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α : Type
    inst✝ : Finite α
    f : α → ι
    b a : α
    j_property : Membership.mem (Set.preimage f (Singleton.singleton (f b))) a
    ⊢ Eq (o.matrixDecomposition (CategoryTheory.CategoryStruct.id (CategoryTheory. …
  -/
  simp only [Set.mem_preimage, Set.mem_singleton_iff] at j_property
  simp only [Category.comp_id, Category.id_comp, Category.assoc, End.one_def, eqToHom_refl,
    Matrix.one_apply, HomOrthogonal.matrixDecomposition_apply, biproduct.components]
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α : Type
    inst✝ : Finite α
    f : α → ι
    b a : α
    j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (f b))) a
    j_property : Eq (f a) (f b)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  split_ifs with h
    /-
      case pos
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α : Type
      inst✝ : Finite α
      f : α → ι
      b a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (f b))) a
      j_property : Eq (f a) (f b)
      h : Eq ⟨b, ⋯⟩ ⟨a, j_property✝⟩
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
  · cases h
    /-
      case pos.refl
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α : Type
      inst✝ : Finite α
      f : α → ι
      b : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (f b))) b
      j_property : Eq (f b) (f b)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α : Type
      inst✝ : Finite α
      f : α → ι
      b a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (f b))) a
      j_property : Eq (f a) (f b)
      h : Not (Eq ⟨b, ⋯⟩ ⟨a, j_property✝⟩)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
  · simp only [Subtype.mk.injEq] at h
    -- Porting note: used to be `convert comp_zero`, but that does not work anymore
    have : biproduct.ι (fun a ↦ s (f a)) a ≫ biproduct.π (fun b ↦ s (f b)) b = 0 := by
      simpa using biproduct.ι_π_ne _ (Ne.symm h)
    /-
      case neg
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α : Type
      inst✝ : Finite α
      f : α → ι
      b a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (f b))) a
      j_property : Eq (f a) (f b)
      h : Not (Eq b a)
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    rw [this, comp_zero]
    /-
      🎉 no goals
    -/


open scoped Classical in
theorem matrixDecomposition_comp (o : HomOrthogonal s) {α β γ : Type} [Finite α] [Fintype β]
    [Finite γ] {f : α → ι} {g : β → ι} {h : γ → ι} (z : (⨁ fun a => s (f a)) ⟶ ⨁ fun b => s (g b))
    (w : (⨁ fun b => s (g b)) ⟶ ⨁ fun c => s (h c)) (i : ι) :
    o.matrixDecomposition (z ≫ w) i = o.matrixDecomposition w i * o.matrixDecomposition z i := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α β γ : Type
    inst✝² : Finite α
    inst✝¹ : Fintype β
    inst✝ : Finite γ
    f : α → ι
    g : β → ι
    h : γ → ι
    z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
    w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
    i : ι
    ⊢ Eq (o.matrixDecomposition (CategoryTheory.CategoryStruct.comp z w) i) (HMul. …
  -/
  ext ⟨c, ⟨⟩⟩ ⟨a, j_property⟩
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α β γ : Type
    inst✝² : Finite α
    inst✝¹ : Fintype β
    inst✝ : Finite γ
    f : α → ι
    g : β → ι
    h : γ → ι
    z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
    w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
    c : γ
    a : α
    j_property : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
    ⊢ Eq (o.matrixDecomposition (CategoryTheory.CategoryStruct.comp z w) (h c) ⟨c, …
  -/
  simp only [Set.mem_preimage, Set.mem_singleton_iff] at j_property
  simp only [Matrix.mul_apply, Limits.biproduct.components,
    HomOrthogonal.matrixDecomposition_apply, Category.comp_id, Category.id_comp, Category.assoc,
    End.mul_def, eqToHom_refl, eqToHom_trans_assoc, Finset.sum_congr]
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α β γ : Type
    inst✝² : Finite α
    inst✝¹ : Fintype β
    inst✝ : Finite γ
    f : α → ι
    g : β → ι
    h : γ → ι
    z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
    w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
    c : γ
    a : α
    j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
    j_property : Eq (f a) (h c)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  conv_lhs => rw [← Category.id_comp w, ← biproduct.total]
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α β γ : Type
    inst✝² : Finite α
    inst✝¹ : Fintype β
    inst✝ : Finite γ
    f : α → ι
    g : β → ι
    h : γ → ι
    z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
    w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
    c : γ
    a : α
    j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
    j_property : Eq (f a) (h c)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp only [Preadditive.sum_comp, Preadditive.comp_sum]
  /-
    case a.mk.refl.mk
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    ι : Type u_1
    s : ι → C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
    o : CategoryTheory.HomOrthogonal s
    α β γ : Type
    inst✝² : Finite α
    inst✝¹ : Fintype β
    inst✝ : Finite γ
    f : α → ι
    g : β → ι
    h : γ → ι
    z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
    w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
    c : γ
    a : α
    j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
    j_property : Eq (f a) (h c)
    ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (CategoryThe …
  -/
  apply Finset.sum_congr_set
    /-
      case a.mk.refl.mk.w
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β γ : Type
      inst✝² : Finite α
      inst✝¹ : Fintype β
      inst✝ : Finite γ
      f : α → ι
      g : β → ι
      h : γ → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
      c : γ
      a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
      j_property : Eq (f a) (h c)
      ⊢ ∀ (x : β) (h_1 : Membership.mem (Set.preimage g (Singleton.singleton (h c))) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.mk.refl.mk.w'
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β γ : Type
      inst✝² : Finite α
      inst✝¹ : Fintype β
      inst✝ : Finite γ
      f : α → ι
      g : β → ι
      h : γ → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
      c : γ
      a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
      j_property : Eq (f a) (h c)
      ⊢ ∀ (x : β), Not (Membership.mem (Set.preimage g (Singleton.singleton (h c)))  …
    -/
  · intro b nm
    /-
      case a.mk.refl.mk.w'
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β γ : Type
      inst✝² : Finite α
      inst✝¹ : Fintype β
      inst✝ : Finite γ
      f : α → ι
      g : β → ι
      h : γ → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
      c : γ
      a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
      j_property : Eq (f a) (h c)
      b : β
      nm : Not (Membership.mem (Set.preimage g (Singleton.singleton (h c))) b)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    simp only [Set.mem_preimage, Set.mem_singleton_iff] at nm
    /-
      case a.mk.refl.mk.w'
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β γ : Type
      inst✝² : Finite α
      inst✝¹ : Fintype β
      inst✝ : Finite γ
      f : α → ι
      g : β → ι
      h : γ → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
      c : γ
      a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
      j_property : Eq (f a) (h c)
      b : β
      nm : Not (Eq (g b) (h c))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    simp only [Category.assoc]
    -- Porting note: this used to be 4 times `convert comp_zero`
    have : biproduct.ι (fun b ↦ s (g b)) b ≫ w ≫ biproduct.π (fun b ↦ s (h b)) c = 0 := by
      apply o.eq_zero nm
    /-
      case a.mk.refl.mk.w'
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      ι : Type u_1
      s : ι → C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasFiniteBiproducts C
      o : CategoryTheory.HomOrthogonal s
      α β γ : Type
      inst✝² : Finite α
      inst✝¹ : Fintype β
      inst✝ : Finite γ
      f : α → ι
      g : β → ι
      h : γ → ι
      z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
      w : Quiver.Hom (CategoryTheory.Limits.biproduct fun b => s (g b)) (CategoryThe …
      c : γ
      a : α
      j_property✝ : Membership.mem (Set.preimage f (Singleton.singleton (h c))) a
      j_property : Eq (f a) (h c)
      b : β
      nm : Not (Eq (g b) (h c))
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    simp only [this, comp_zero]
    /-
      🎉 no goals
    -/


/-- `HomOrthogonal.MatrixDecomposition` as an `R`-linear equivalence. -/
@[simps]
noncomputable def matrixDecompositionLinearEquiv (o : HomOrthogonal s) {α β : Type} [Finite α]
    [Finite β] {f : α → ι} {g : β → ι} :
    ((⨁ fun a => s (f a)) ⟶ ⨁ fun b => s (g b)) ≃ₗ[R]
      ∀ i : ι, Matrix (g ⁻¹' {i}) (f ⁻¹' {i}) (End (s i)) :=
  { o.matrixDecompositionAddEquiv with
    map_smul' := fun w z => by
      /-
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Limits.HasFiniteBiproducts C
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : CategoryTheory.Linear R C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w : R
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul w z)) (HSMul …
      -/
      ext
      /-
        case h.a
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Limits.HasFiniteBiproducts C
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : CategoryTheory.Linear R C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w : R
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        x✝ : ι
        i✝ : ↑(Set.preimage g (Singleton.singleton x✝))
        j✝ : ↑(Set.preimage f (Singleton.singleton x✝))
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul w z) x✝ i✝ j …
      -/
      dsimp [biproduct.components]
      /-
        case h.a
        C : Type u
        inst✝⁶ : CategoryTheory.Category.{v, u} C
        ι : Type u_1
        s : ι → C
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Limits.HasFiniteBiproducts C
        R : Type u_2
        inst✝³ : Semiring R
        inst✝² : CategoryTheory.Linear R C
        o : CategoryTheory.HomOrthogonal s
        α β : Type
        inst✝¹ : Finite α
        inst✝ : Finite β
        f : α → ι
        g : β → ι
        w : R
        z : Quiver.Hom (CategoryTheory.Limits.biproduct fun a => s (f a)) (CategoryThe …
        x✝ : ι
        i✝ : ↑(Set.preimage g (Singleton.singleton x✝))
        j✝ : ↑(Set.preimage f (Singleton.singleton x✝))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp }
      /-
        🎉 no goals
      -/


/-- Given a hom orthogonal family `s : ι → C`
for which each `End (s i)` is a ring with invariant basis number (e.g. if each `s i` is simple),
if two direct sums over `s` are isomorphic, then they have the same multiplicities.
-/
theorem equiv_of_iso (o : HomOrthogonal s) {α β : Type} [Finite α] [Finite β] {f : α → ι}
    {g : β → ι} (i : (⨁ fun a => s (f a)) ≅ ⨁ fun b => s (g b)) :
    ∃ e : α ≃ β, ∀ a, g (e a) = f a := by
  classical
  refine ⟨Equiv.ofPreimageEquiv ?_, fun a => Equiv.ofPreimageEquiv_map _ _⟩
  intro c
  apply Nonempty.some
  apply Cardinal.eq.1
  cases nonempty_fintype α; cases nonempty_fintype β
  simp only [Cardinal.mk_fintype, Nat.cast_inj]
  exact
    Matrix.square_of_invertible (o.matrixDecomposition i.inv c) (o.matrixDecomposition i.hom c)
      (by
        rw [← o.matrixDecomposition_comp]
        simp)
      (by
        rw [← o.matrixDecomposition_comp]
        simp)


