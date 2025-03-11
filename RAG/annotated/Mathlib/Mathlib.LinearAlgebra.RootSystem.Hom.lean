/-- A morphism of root pairings is a pair of mutually transposed maps of weight and coweight spaces
that preserves roots and coroots.  We make the map of indexing sets explicit. -/
@[ext]
structure Hom {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) where
  /-- A linear map on weight space. -/
  weightMap : M →ₗ[R] M₂
  /-- A contravariant linear map on coweight space. -/
  coweightMap : N₂ →ₗ[R] N
  /-- A bijection on index sets. -/
  indexEquiv : ι ≃ ι₂
  weight_coweight_transpose : weightMap.dualMap ∘ₗ Q.toDualRight = P.toDualRight ∘ₗ coweightMap
  root_weightMap : weightMap ∘ P.root = Q.root ∘ indexEquiv
  coroot_coweightMap : coweightMap ∘ Q.coroot = P.coroot ∘ indexEquiv.symm


lemma weight_coweight_transpose_apply {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (x : N₂) (f : Hom P Q) :
    f.weightMap.dualMap (Q.toDualRight x) = P.toDualRight (f.coweightMap x) :=
  Eq.mp (propext LinearMap.ext_iff) f.weight_coweight_transpose x


lemma root_weightMap_apply {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (i : ι) (f : Hom P Q) :
    f.weightMap (P.root i) = Q.root (f.indexEquiv i) :=
  Eq.mp (propext funext_iff) f.root_weightMap i


lemma coroot_coweightMap_apply {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (i : ι₂) (f : Hom P Q) :
    f.coweightMap (Q.coroot i) = P.coroot (f.indexEquiv.symm i) :=
  Eq.mp (propext funext_iff) f.coroot_coweightMap i


/-- The identity morphism of a root pairing. -/
@[simps!]
def id (P : RootPairing ι R M N) : Hom P P where
  weightMap := LinearMap.id
  coweightMap := LinearMap.id
  indexEquiv := Equiv.refl ι
                                  /-
                                    ι : Type u_1
                                    R : Type u_2
                                    M : Type u_3
                                    N : Type u_4
                                    inst✝⁴ : CommRing R
                                    inst✝³ : AddCommGroup M
                                    inst✝² : Module R M
                                    inst✝¹ : AddCommGroup N
                                    inst✝ : Module R N
                                    P : RootPairing ι R M N
                                    ⊢ Eq (LinearMap.id.dualMap.comp ↑P.toDualRight) ((↑P.toDualRight).comp LinearM …
                                  -/
  weight_coweight_transpose := by simp
                                  /-
                                    🎉 no goals
                                  -/
                       /-
                         ι : Type u_1
                         R : Type u_2
                         M : Type u_3
                         N : Type u_4
                         inst✝⁴ : CommRing R
                         inst✝³ : AddCommGroup M
                         inst✝² : Module R M
                         inst✝¹ : AddCommGroup N
                         inst✝ : Module R N
                         P : RootPairing ι R M N
                         ⊢ Eq (Function.comp ⇑LinearMap.id ⇑P.root) (Function.comp ⇑P.root ⇑(Equiv.refl …
                       -/
  root_weightMap := by simp
                       /-
                         🎉 no goals
                       -/
                           /-
                             ι : Type u_1
                             R : Type u_2
                             M : Type u_3
                             N : Type u_4
                             inst✝⁴ : CommRing R
                             inst✝³ : AddCommGroup M
                             inst✝² : Module R M
                             inst✝¹ : AddCommGroup N
                             inst✝ : Module R N
                             P : RootPairing ι R M N
                             ⊢ Eq (Function.comp ⇑LinearMap.id ⇑P.coroot) (Function.comp ⇑P.coroot ⇑(Equiv. …
                           -/
  coroot_coweightMap := by simp
                           /-
                             🎉 no goals
                           -/


/-- Composition of morphisms -/
@[simps!]
def comp {ι₁ M₁ N₁ ι₂ M₂ N₂ : Type*} [AddCommGroup M₁] [Module R M₁] [AddCommGroup N₁]
    [Module R N₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    {P : RootPairing ι R M N} {P₁ : RootPairing ι₁ R M₁ N₁} {P₂ : RootPairing ι₂ R M₂ N₂}
    (g : Hom P₁ P₂) (f : Hom P P₁) : Hom P P₂ where
  weightMap := g.weightMap ∘ₗ f.weightMap
  coweightMap := f.coweightMap ∘ₗ g.coweightMap
  indexEquiv := f.indexEquiv.trans g.indexEquiv
  weight_coweight_transpose := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      ⊢ Eq ((g.weightMap.comp f.weightMap).dualMap.comp ↑P₂.toDualRight) ((↑P.toDual …
    -/
    ext φ x
    rw [← LinearMap.dualMap_comp_dualMap, ← LinearMap.comp_assoc _ f.coweightMap,
      ← f.weight_coweight_transpose, LinearMap.comp_assoc g.coweightMap,
      ← g.weight_coweight_transpose, ← LinearMap.comp_assoc]
  root_weightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      ⊢ Eq (Function.comp ⇑(g.weightMap.comp f.weightMap) ⇑P.root) (Function.comp ⇑P …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      i : ι
      ⊢ Eq (Function.comp (⇑(g.weightMap.comp f.weightMap)) (⇑P.root) i) (Function.c …
    -/
    simp only [LinearMap.coe_comp, Equiv.coe_trans]
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      i : ι
      ⊢ Eq (Function.comp (Function.comp ⇑g.weightMap ⇑f.weightMap) (⇑P.root) i) (Fu …
    -/
    rw [comp_assoc, f.root_weightMap, ← comp_assoc, g.root_weightMap, comp_assoc]
    /-
      🎉 no goals
    -/
  coroot_coweightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      ⊢ Eq (Function.comp ⇑(f.coweightMap.comp g.coweightMap) ⇑P₂.coroot) (Function. …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      i : ι₂
      ⊢ Eq (Function.comp (⇑(f.coweightMap.comp g.coweightMap)) (⇑P₂.coroot) i) (Fun …
    -/
    simp only [LinearMap.coe_comp, Equiv.symm_trans_apply]
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      i : ι₂
      ⊢ Eq (Function.comp (Function.comp ⇑f.coweightMap ⇑g.coweightMap) (⇑P₂.coroot) …
    -/
    rw [comp_assoc, g.coroot_coweightMap, ← comp_assoc, f.coroot_coweightMap, comp_assoc]
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₁ : Type u_5
      M₁ : Type u_6
      N₁ : Type u_7
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module R M₁
      inst✝⁵ : AddCommGroup N₁
      inst✝⁴ : Module R N₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      P₁ : RootPairing ι₁ R M₁ N₁
      P₂ : RootPairing ι₂ R M₂ N₂
      g : P₁.Hom P₂
      f : P.Hom P₁
      i : ι₂
      ⊢ Eq (Function.comp (⇑P.coroot) (Function.comp ⇑f.indexEquiv.symm ⇑g.indexEqui …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
lemma id_comp {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (f : Hom P Q) :
    comp f (id P) = f := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    f : P.Hom Q
    ⊢ Eq (f.comp (RootPairing.Hom.id P)) f
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  ext x <;> simp
            /-
              🎉 no goals
            -/


@[simp]
lemma comp_id {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (f : Hom P Q) :
    comp (id Q) f = f := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    f : P.Hom Q
    ⊢ Eq ((RootPairing.Hom.id Q).comp f) f
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  ext x <;> simp
            /-
              🎉 no goals
            -/


@[simp]
lemma comp_assoc {ι₁ M₁ N₁ ι₂ M₂ N₂ ι₃ M₃ N₃ : Type*} [AddCommGroup M₁] [Module R M₁]
    [AddCommGroup N₁] [Module R N₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    [AddCommGroup M₃] [Module R M₃] [AddCommGroup N₃] [Module R N₃] {P : RootPairing ι R M N}
    {P₁ : RootPairing ι₁ R M₁ N₁} {P₂ : RootPairing ι₂ R M₂ N₂} {P₃ : RootPairing ι₃ R M₃ N₃}
    (h : Hom P₂ P₃) (g : Hom P₁ P₂) (f : Hom P P₁) :
    comp (comp h g) f = comp h (comp g f) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    ι₁ : Type u_5
    M₁ : Type u_6
    N₁ : Type u_7
    ι₂ : Type u_8
    M₂ : Type u_9
    N₂ : Type u_10
    ι₃ : Type u_11
    M₃ : Type u_12
    N₃ : Type u_13
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup N₁
    inst✝⁸ : Module R N₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    P : RootPairing ι R M N
    P₁ : RootPairing ι₁ R M₁ N₁
    P₂ : RootPairing ι₂ R M₂ N₂
    P₃ : RootPairing ι₃ R M₃ N₃
    h : P₂.Hom P₃
    g : P₁.Hom P₂
    f : P.Hom P₁
    ⊢ Eq ((h.comp g).comp f) (h.comp (g.comp f))
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- The endomorphism monoid of a root pairing. -/
instance (P : RootPairing ι R M N) : Monoid (Hom P P) where
  mul := comp
  mul_assoc := comp_assoc
  one := id P
  one_mul := id_comp P P
  mul_one := comp_id P P


@[simp]
lemma weightMap_one (P : RootPairing ι R M N) :
    weightMap (P := P) (Q := P) 1 = LinearMap.id (R := R) (M := M) :=
  rfl


@[simp]
lemma coweightMap_one (P : RootPairing ι R M N) :
    coweightMap (P := P) (Q := P) 1 = LinearMap.id (R := R) (M := N) :=
  rfl


@[simp]
lemma indexEquiv_one (P : RootPairing ι R M N) :
    indexEquiv (P := P) (Q := P) 1 = Equiv.refl ι :=
  rfl


@[simp]
lemma weightMap_mul (P : RootPairing ι R M N) (x y : Hom P P) :
    weightMap (x * y) = weightMap x ∘ₗ weightMap y :=
  rfl


@[simp]
lemma coweightMap_mul (P : RootPairing ι R M N) (x y : Hom P P) :
    coweightMap (x * y) = coweightMap y ∘ₗ coweightMap x :=
  rfl


@[simp]
lemma indexEquiv_mul (P : RootPairing ι R M N) (x y : Hom P P) :
    indexEquiv (x * y) = indexEquiv x ∘ indexEquiv y :=
  rfl


/-- The endomorphism monoid of a root pairing. -/
abbrev _root_.RootPairing.End (P : RootPairing ι R M N) := Hom P P


/-- The weight space representation of endomorphisms -/
def weightHom (P : RootPairing ι R M N) : End P →* (Module.End R M) where
  toFun g := Hom.weightMap (P := P) (Q := P) g
                     /-
                       ι : Type u_1
                       R : Type u_2
                       M : Type u_3
                       N : Type u_4
                       inst✝⁴ : CommRing R
                       inst✝³ : AddCommGroup M
                       inst✝² : Module R M
                       inst✝¹ : AddCommGroup N
                       inst✝ : Module R N
                       P : RootPairing ι R M N
                       g h : P.End
                       ⊢ Eq ({ toFun := fun g => g.weightMap, map_one' := ⋯ }.toFun (HMul.hMul g h))  …
                     -/
                 /-
                   ι : Type u_1
                   R : Type u_2
                   M : Type u_3
                   N : Type u_4
                   inst✝⁴ : CommRing R
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   inst✝¹ : AddCommGroup N
                   inst✝ : Module R N
                   P : RootPairing ι R M N
                   ⊢ Eq ((fun g => g.weightMap) 1) 1
                 -/
  map_mul' g h := by ext; simp
                      /-
                        🎉 no goals
                      -/
                          /-
                            🎉 no goals
                          -/
  map_one' := by ext; simp


lemma weightHom_injective (P : RootPairing ι R M N) : Injective (weightHom P) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Function.Injective ⇑(RootPairing.Hom.weightHom P)
  -/
  intro f g hfg
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    f g : P.End
    hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
    ⊢ Eq f g
  -/
  ext x
    /-
      case weightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : M
      ⊢ Eq (f.weightMap x) (g.weightMap x)
    -/
  · exact LinearMap.congr_fun hfg x
    /-
      🎉 no goals
    -/
    /-
      case coweightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : N
      ⊢ Eq (f.coweightMap x) (g.coweightMap x)
    -/
  · refine LinearEquiv.injective P.toDualRight ?_
    /-
      case coweightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : N
      ⊢ Eq (P.toDualRight (f.coweightMap x)) (P.toDualRight (g.coweightMap x))
    -/
    simp_rw [← weight_coweight_transpose_apply]
    /-
      case coweightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : N
      ⊢ Eq (f.weightMap.dualMap (P.toDualRight x)) (g.weightMap.dualMap (P.toDualRig …
    -/
    exact congrFun (congrArg DFunLike.coe (congrArg LinearMap.dualMap hfg)) (P.toDualRight x)
    /-
      🎉 no goals
    -/
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : ι
      ⊢ Eq (f.indexEquiv x) (g.indexEquiv x)
    -/
  · refine Embedding.injective P.root ?_
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : ι
      ⊢ Eq (P.root (f.indexEquiv x)) (P.root (g.indexEquiv x))
    -/
    simp_rw [← root_weightMap_apply]
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.weightHom P) f) ((RootPairing.Hom.weightHom P) g)
      x : ι
      ⊢ Eq (f.weightMap (P.root x)) (g.weightMap (P.root x))
    -/
    exact congrFun (congrArg DFunLike.coe hfg) (P.root x)
    /-
      🎉 no goals
    -/


/-- The coweight space representation of endomorphisms -/
def coweightHom (P : RootPairing ι R M N) : End P →* (N →ₗ[R] N)ᵐᵒᵖ where
  toFun g := MulOpposite.op (Hom.coweightMap (P := P) (Q := P) g)
  map_mul' g h := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      g h : P.End
      ⊢ Eq ({ toFun := fun g => MulOpposite.op g.coweightMap, map_one' := ⋯ }.toFun  …
    -/
    simp only [← MulOpposite.op_mul, coweightMap_mul, LinearMap.mul_eq_comp]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      ⊢ Eq ((fun g => MulOpposite.op g.coweightMap) 1) 1
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_one' := by
    simp only [MulOpposite.op_eq_one_iff, coweightMap_one, LinearMap.one_eq_id]


lemma coweightHom_injective (P : RootPairing ι R M N) : Injective (coweightHom P) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Function.Injective ⇑(RootPairing.Hom.coweightHom P)
  -/
  intro f g hfg
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    f g : P.End
    hfg : Eq ((RootPairing.Hom.coweightHom P) f) ((RootPairing.Hom.coweightHom P) g)
    ⊢ Eq f g
  -/
  ext x
    /-
      case weightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.coweightHom P) f) ((RootPairing.Hom.coweightHom P) g)
      x : M
      ⊢ Eq (f.weightMap x) (g.weightMap x)
    -/
  · dsimp [coweightHom] at hfg
    /-
      case weightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq (MulOpposite.op f.coweightMap) (MulOpposite.op g.coweightMap)
      x : M
      ⊢ Eq (f.weightMap x) (g.weightMap x)
    -/
    rw [MulOpposite.op_inj] at hfg
    have h := congrArg (LinearMap.comp (M₃ := Module.Dual R M)
        (σ₂₃ := RingHom.id R) (P.toDualRight)) hfg
    /-
      case weightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : M
      h : Eq ((↑P.toDualRight).comp f.coweightMap) ((↑P.toDualRight).comp g.coweight …
      ⊢ Eq (f.weightMap x) (g.weightMap x)
    -/
    rw [← f.weight_coweight_transpose, ← g.weight_coweight_transpose] at h
    have : f.weightMap = g.weightMap := by
      haveI : Module.IsReflexive R M := PerfectPairing.reflexive_left P.toPerfectPairing
      refine (Module.dualMap_dualMap_eq_iff R M).mp (congrArg LinearMap.dualMap
        ((LinearEquiv.eq_comp_toLinearMap_iff f.weightMap.dualMap g.weightMap.dualMap).mp h))
    /-
      case weightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : M
      h : Eq (f.weightMap.dualMap.comp ↑P.toDualRight) (g.weightMap.dualMap.comp ↑P. …
      this : Eq f.weightMap g.weightMap
      ⊢ Eq (f.weightMap x) (g.weightMap x)
    -/
    exact congrFun (congrArg DFunLike.coe this) x
    /-
      🎉 no goals
    -/
    /-
      case coweightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.coweightHom P) f) ((RootPairing.Hom.coweightHom P) g)
      x : N
      ⊢ Eq (f.coweightMap x) (g.coweightMap x)
    -/
  · dsimp [coweightHom] at hfg
    /-
      case coweightMap.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq (MulOpposite.op f.coweightMap) (MulOpposite.op g.coweightMap)
      x : N
      ⊢ Eq (f.coweightMap x) (g.coweightMap x)
    -/
    simp_all only [MulOpposite.op_inj]
    /-
      🎉 no goals
    -/
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq ((RootPairing.Hom.coweightHom P) f) ((RootPairing.Hom.coweightHom P) g)
      x : ι
      ⊢ Eq (f.indexEquiv x) (g.indexEquiv x)
    -/
  · dsimp [coweightHom] at hfg
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq (MulOpposite.op f.coweightMap) (MulOpposite.op g.coweightMap)
      x : ι
      ⊢ Eq (f.indexEquiv x) (g.indexEquiv x)
    -/
    rw [MulOpposite.op_inj] at hfg
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : ι
      ⊢ Eq (f.indexEquiv x) (g.indexEquiv x)
    -/
    set y := f.indexEquiv x with hy
    have : f.coweightMap (P.coroot y) = g.coweightMap (P.coroot y) := by
      exact congrFun (congrArg DFunLike.coe hfg) (P.coroot y)
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : ι
      y : ι := f.indexEquiv x
      hy : Eq y (f.indexEquiv x)
      this : Eq (f.coweightMap (P.coroot y)) (g.coweightMap (P.coroot y))
      ⊢ Eq y (g.indexEquiv x)
    -/
    rw [coroot_coweightMap_apply, coroot_coweightMap_apply, Embedding.apply_eq_iff_eq, hy] at this
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : ι
      y : ι := f.indexEquiv x
      hy : Eq y (f.indexEquiv x)
      this : Eq (f.indexEquiv.symm (f.indexEquiv x)) (g.indexEquiv.symm (f.indexEqui …
      ⊢ Eq y (g.indexEquiv x)
    -/
    rw [Equiv.symm_apply_apply] at this
    /-
      case indexEquiv.H
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      f g : P.End
      hfg : Eq f.coweightMap g.coweightMap
      x : ι
      y : ι := f.indexEquiv x
      hy : Eq y (f.indexEquiv x)
      this : Eq x (g.indexEquiv.symm (f.indexEquiv x))
      ⊢ Eq y (g.indexEquiv x)
    -/
    rw [this, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


/-- The permutation representation of the endomorphism monoid on the root index set -/
def indexHom (P : RootPairing ι R M N) : End P →* (ι ≃ ι) where
  toFun f := Hom.indexEquiv f
                 /-
                   ι : Type u_1
                   R : Type u_2
                   M : Type u_3
                   N : Type u_4
                   inst✝⁴ : CommRing R
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   inst✝¹ : AddCommGroup N
                   inst✝ : Module R N
                   P : RootPairing ι R M N
                   ⊢ Eq ((fun f => f.indexEquiv) 1) 1
                 -/
  map_one' := by ext; simp
                      /-
                        🎉 no goals
                      -/
                     /-
                       ι : Type u_1
                       R : Type u_2
                       M : Type u_3
                       N : Type u_4
                       inst✝⁴ : CommRing R
                       inst✝³ : AddCommGroup M
                       inst✝² : Module R M
                       inst✝¹ : AddCommGroup N
                       inst✝ : Module R N
                       P : RootPairing ι R M N
                       x y : P.End
                       ⊢ Eq ({ toFun := fun f => f.indexEquiv, map_one' := ⋯ }.toFun (HMul.hMul x y)) …
                     -/
  map_mul' x y := by ext; simp
                          /-
                            🎉 no goals
                          -/


/-- An equivalence of root pairings is a morphism where the maps of weight and coweight spaces are
bijective.

See also `RootPairing.Equiv.toEndUnit`. -/
@[ext]
protected structure Equiv extends Hom P Q where
  bijective_weightMap : Bijective weightMap
  bijective_coweightMap : Bijective coweightMap


/-- The linear equivalence of weight spaces given by an equivalence of root pairings. -/
def weightEquiv (e : RootPairing.Equiv P Q) : M ≃ₗ[R] M₂ :=
    LinearEquiv.ofBijective _ e.bijective_weightMap


@[simp]
lemma weightEquiv_apply (e : RootPairing.Equiv P Q) (m : M) :
    weightEquiv P Q e m = e.toHom.weightMap m :=
  rfl


@[simp]
lemma weightEquiv_symm_weightMap (e : RootPairing.Equiv P Q) (m : M) :
    (weightEquiv P Q e).symm (e.toHom.weightMap m) = m :=
  (LinearEquiv.symm_apply_eq (weightEquiv P Q e)).mpr rfl


@[simp]
lemma weightMap_weightEquiv_symm (e : RootPairing.Equiv P Q) (m : M₂) :
    e.toHom.weightMap ((weightEquiv P Q e).symm m) = m := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    e : P.Equiv Q
    m : M₂
    ⊢ Eq ((↑e).weightMap ((RootPairing.Equiv.weightEquiv P Q e).symm m)) m
  -/
  rw [← weightEquiv_apply]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    e : P.Equiv Q
    m : M₂
    ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q e) ((RootPairing.Equiv.weightEquiv P  …
  -/
  exact LinearEquiv.apply_symm_apply (weightEquiv P Q e) m
  /-
    🎉 no goals
  -/


/-- The contravariant equivalence of coweight spaces given by an equivalence of root pairings. -/
def coweightEquiv (e : RootPairing.Equiv P Q) : N₂ ≃ₗ[R] N :=
  LinearEquiv.ofBijective _ e.bijective_coweightMap


@[simp]
lemma coweightEquiv_apply (e : RootPairing.Equiv P Q) (n : N₂) :
    coweightEquiv P Q e n = e.toHom.coweightMap n :=
  rfl


@[simp]
lemma coweightEquiv_symm_coweightMap (e : RootPairing.Equiv P Q) (n : N₂) :
    (coweightEquiv P Q e).symm (e.toHom.coweightMap n) = n :=
  (LinearEquiv.symm_apply_eq (coweightEquiv P Q e)).mpr rfl


@[simp]
lemma coweightMap_coweightEquiv_symm (e : RootPairing.Equiv P Q) (n : N) :
    e.toHom.coweightMap ((coweightEquiv P Q e).symm n) = n := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    e : P.Equiv Q
    n : N
    ⊢ Eq ((↑e).coweightMap ((RootPairing.Equiv.coweightEquiv P Q e).symm n)) n
  -/
  rw [← coweightEquiv_apply]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_5
    M₂ : Type u_6
    N₂ : Type u_7
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    e : P.Equiv Q
    n : N
    ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q e) ((RootPairing.Equiv.coweightEqui …
  -/
  exact LinearEquiv.apply_symm_apply (coweightEquiv P Q e) n
  /-
    🎉 no goals
  -/


/-- The identity equivalence of a root pairing. -/
@[simps!]
def id (P : RootPairing ι R M N) : RootPairing.Equiv P P :=
  { Hom.id P with
    bijective_weightMap := _root_.id bijective_id
    bijective_coweightMap := _root_.id bijective_id }


/-- Composition of equivalences -/
@[simps!]
def comp {ι₁ M₁ N₁ ι₂ M₂ N₂ : Type*} [AddCommGroup M₁] [Module R M₁] [AddCommGroup N₁]
    [Module R N₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    {P : RootPairing ι R M N} {P₁ : RootPairing ι₁ R M₁ N₁} {P₂ : RootPairing ι₂ R M₂ N₂}
    (g : RootPairing.Equiv P₁ P₂) (f : RootPairing.Equiv P P₁) : RootPairing.Equiv P P₂ :=
  { Hom.comp g.toHom f.toHom with
    bijective_weightMap := by
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁶ : CommRing R
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : AddCommGroup N
        inst✝¹² : Module R N
        ι₂✝ : Type u_5
        M₂✝ : Type u_6
        N₂✝ : Type u_7
        inst✝¹¹ : AddCommGroup M₂✝
        inst✝¹⁰ : Module R M₂✝
        inst✝⁹ : AddCommGroup N₂✝
        inst✝⁸ : Module R N₂✝
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂✝ R M₂✝ N₂✝
        ι₁ : Type u_8
        M₁ : Type u_9
        N₁ : Type u_10
        ι₂ : Type u_11
        M₂ : Type u_12
        N₂ : Type u_13
        inst✝⁷ : AddCommGroup M₁
        inst✝⁶ : Module R M₁
        inst✝⁵ : AddCommGroup N₁
        inst✝⁴ : Module R N₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P : RootPairing ι R M N
        P₁ : RootPairing ι₁ R M₁ N₁
        P₂ : RootPairing ι₂ R M₂ N₂
        g : P₁.Equiv P₂
        f : P.Equiv P₁
        ⊢ Function.Bijective ⇑__src✝.weightMap
      -/
      simp only [Hom.comp, LinearMap.coe_comp]
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁶ : CommRing R
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : AddCommGroup N
        inst✝¹² : Module R N
        ι₂✝ : Type u_5
        M₂✝ : Type u_6
        N₂✝ : Type u_7
        inst✝¹¹ : AddCommGroup M₂✝
        inst✝¹⁰ : Module R M₂✝
        inst✝⁹ : AddCommGroup N₂✝
        inst✝⁸ : Module R N₂✝
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂✝ R M₂✝ N₂✝
        ι₁ : Type u_8
        M₁ : Type u_9
        N₁ : Type u_10
        ι₂ : Type u_11
        M₂ : Type u_12
        N₂ : Type u_13
        inst✝⁷ : AddCommGroup M₁
        inst✝⁶ : Module R M₁
        inst✝⁵ : AddCommGroup N₁
        inst✝⁴ : Module R N₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P : RootPairing ι R M N
        P₁ : RootPairing ι₁ R M₁ N₁
        P₂ : RootPairing ι₂ R M₂ N₂
        g : P₁.Equiv P₂
        f : P.Equiv P₁
        ⊢ Function.Bijective (Function.comp ⇑(↑g).weightMap ⇑(↑f).weightMap)
      -/
      exact Bijective.comp g.bijective_weightMap f.bijective_weightMap
      /-
        🎉 no goals
      -/
    bijective_coweightMap := by
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁶ : CommRing R
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : AddCommGroup N
        inst✝¹² : Module R N
        ι₂✝ : Type u_5
        M₂✝ : Type u_6
        N₂✝ : Type u_7
        inst✝¹¹ : AddCommGroup M₂✝
        inst✝¹⁰ : Module R M₂✝
        inst✝⁹ : AddCommGroup N₂✝
        inst✝⁸ : Module R N₂✝
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂✝ R M₂✝ N₂✝
        ι₁ : Type u_8
        M₁ : Type u_9
        N₁ : Type u_10
        ι₂ : Type u_11
        M₂ : Type u_12
        N₂ : Type u_13
        inst✝⁷ : AddCommGroup M₁
        inst✝⁶ : Module R M₁
        inst✝⁵ : AddCommGroup N₁
        inst✝⁴ : Module R N₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P : RootPairing ι R M N
        P₁ : RootPairing ι₁ R M₁ N₁
        P₂ : RootPairing ι₂ R M₂ N₂
        g : P₁.Equiv P₂
        f : P.Equiv P₁
        ⊢ Function.Bijective ⇑__src✝.coweightMap
      -/
      simp only [Hom.comp, LinearMap.coe_comp]
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁶ : CommRing R
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : AddCommGroup N
        inst✝¹² : Module R N
        ι₂✝ : Type u_5
        M₂✝ : Type u_6
        N₂✝ : Type u_7
        inst✝¹¹ : AddCommGroup M₂✝
        inst✝¹⁰ : Module R M₂✝
        inst✝⁹ : AddCommGroup N₂✝
        inst✝⁸ : Module R N₂✝
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂✝ R M₂✝ N₂✝
        ι₁ : Type u_8
        M₁ : Type u_9
        N₁ : Type u_10
        ι₂ : Type u_11
        M₂ : Type u_12
        N₂ : Type u_13
        inst✝⁷ : AddCommGroup M₁
        inst✝⁶ : Module R M₁
        inst✝⁵ : AddCommGroup N₁
        inst✝⁴ : Module R N₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P : RootPairing ι R M N
        P₁ : RootPairing ι₁ R M₁ N₁
        P₂ : RootPairing ι₂ R M₂ N₂
        g : P₁.Equiv P₂
        f : P.Equiv P₁
        ⊢ Function.Bijective (Function.comp ⇑(↑f).coweightMap ⇑(↑g).coweightMap)
      -/
      exact Bijective.comp f.bijective_coweightMap g.bijective_coweightMap }
      /-
        🎉 no goals
      -/


@[simp]
lemma toHom_comp {ι₁ M₁ N₁ ι₂ M₂ N₂ : Type*} [AddCommGroup M₁] [Module R M₁] [AddCommGroup N₁]
    [Module R N₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    {P : RootPairing ι R M N} {P₁ : RootPairing ι₁ R M₁ N₁} {P₂ : RootPairing ι₂ R M₂ N₂}
    (g : RootPairing.Equiv P₁ P₂) (f : RootPairing.Equiv P P₁) :
    (Equiv.comp g f).toHom = Hom.comp g.toHom f.toHom := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : AddCommGroup N
    inst✝⁸ : Module R N
    ι₁ : Type u_8
    M₁ : Type u_9
    N₁ : Type u_10
    ι₂ : Type u_11
    M₂ : Type u_12
    N₂ : Type u_13
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    P₁ : RootPairing ι₁ R M₁ N₁
    P₂ : RootPairing ι₂ R M₂ N₂
    g : P₁.Equiv P₂
    f : P.Equiv P₁
    ⊢ Eq (↑(g.comp f)) ((↑g).comp ↑f)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma id_comp {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (f : RootPairing.Equiv P Q) :
    comp f (id P) = f := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_8
    M₂ : Type u_9
    N₂ : Type u_10
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    f : P.Equiv Q
    ⊢ Eq (f.comp (RootPairing.Equiv.id P)) f
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  ext x <;> simp
            /-
              🎉 no goals
            -/


@[simp]
lemma comp_id {ι₂ M₂ N₂ : Type*}
    [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (f : RootPairing.Equiv P Q) :
    comp (id Q) f = f := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    ι₂ : Type u_8
    M₂ : Type u_9
    N₂ : Type u_10
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommGroup N₂
    inst✝ : Module R N₂
    P : RootPairing ι R M N
    Q : RootPairing ι₂ R M₂ N₂
    f : P.Equiv Q
    ⊢ Eq ((RootPairing.Equiv.id Q).comp f) f
  -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  ext x <;> simp
            /-
              🎉 no goals
            -/


@[simp]
lemma comp_assoc {ι₁ M₁ N₁ ι₂ M₂ N₂ ι₃ M₃ N₃ : Type*} [AddCommGroup M₁] [Module R M₁]
    [AddCommGroup N₁] [Module R N₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    [AddCommGroup M₃] [Module R M₃] [AddCommGroup N₃] [Module R N₃] {P : RootPairing ι R M N}
    {P₁ : RootPairing ι₁ R M₁ N₁} {P₂ : RootPairing ι₂ R M₂ N₂} {P₃ : RootPairing ι₃ R M₃ N₃}
    (h : RootPairing.Equiv P₂ P₃) (g : RootPairing.Equiv P₁ P₂) (f : RootPairing.Equiv P P₁) :
    comp (comp h g) f = comp h (comp g f) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹⁶ : CommRing R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Module R M
    inst✝¹³ : AddCommGroup N
    inst✝¹² : Module R N
    ι₁ : Type u_8
    M₁ : Type u_9
    N₁ : Type u_10
    ι₂ : Type u_11
    M₂ : Type u_12
    N₂ : Type u_13
    ι₃ : Type u_14
    M₃ : Type u_15
    N₃ : Type u_16
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup N₁
    inst✝⁸ : Module R N₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : Module R M₂
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    inst✝³ : AddCommGroup M₃
    inst✝² : Module R M₃
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    P : RootPairing ι R M N
    P₁ : RootPairing ι₁ R M₁ N₁
    P₂ : RootPairing ι₂ R M₂ N₂
    P₃ : RootPairing ι₃ R M₃ N₃
    h : P₂.Equiv P₃
    g : P₁.Equiv P₂
    f : P.Equiv P₁
    ⊢ Eq ((h.comp g).comp f) (h.comp (g.comp f))
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- Equivalences form a monoid. -/
instance (P : RootPairing ι R M N) : Monoid (RootPairing.Equiv P P) where
  mul := comp
  mul_assoc := comp_assoc
  one := id P
  one_mul := id_comp P P
  mul_one := comp_id P P


@[simp]
lemma weightEquiv_one (P : RootPairing ι R M N) :
    weightEquiv (P := P) (Q := P) 1 = LinearMap.id (R := R) (M := M) :=
  rfl


@[simp]
lemma coweightEquiv_one (P : RootPairing ι R M N) :
    coweightEquiv (P := P) (Q := P) 1 = LinearMap.id (R := R) (M := N) :=
  rfl


@[simp]
lemma toHom_one (P : RootPairing ι R M N) :
    (1 : RootPairing.Equiv P P).toHom = (1 : RootPairing.Hom P P) :=
  rfl


@[simp]
lemma mul_eq_comp {P : RootPairing ι R M N} (x y : RootPairing.Equiv P P) :
    x * y = Equiv.comp x y :=
  rfl


@[simp]
lemma weightEquiv_comp_toLin {P : RootPairing ι R M N} (x y : RootPairing.Equiv P P) :
    weightEquiv P P (Equiv.comp x y) = weightEquiv P P y ≪≫ₗ weightEquiv P P x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : P.Equiv P
    ⊢ Eq (RootPairing.Equiv.weightEquiv P P (x.comp y)) ((RootPairing.Equiv.weight …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma weightEquiv_mul {P : RootPairing ι R M N} (x y : RootPairing.Equiv P P) :
    weightEquiv P P x * weightEquiv P P y = weightEquiv P P y ≪≫ₗ weightEquiv P P x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : P.Equiv P
    ⊢ Eq (HMul.hMul (RootPairing.Equiv.weightEquiv P P x) (RootPairing.Equiv.weigh …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma coweightEquiv_comp_toLin {P : RootPairing ι R M N} (x y : RootPairing.Equiv P P) :
    coweightEquiv P P (Equiv.comp x y) = coweightEquiv P P x ≪≫ₗ coweightEquiv P P y := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : P.Equiv P
    ⊢ Eq (RootPairing.Equiv.coweightEquiv P P (x.comp y)) ((RootPairing.Equiv.cowe …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma coweightEquiv_mul {P : RootPairing ι R M N} (x y : RootPairing.Equiv P P) :
    coweightEquiv P P x * coweightEquiv P P y = coweightEquiv P P y ≪≫ₗ coweightEquiv P P x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    x y : P.Equiv P
    ⊢ Eq (HMul.hMul (RootPairing.Equiv.coweightEquiv P P x) (RootPairing.Equiv.cow …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The inverse of a root pairing equivalence. -/
def symm {ι₂ M₂ N₂ : Type*} [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂] [Module R N₂]
    (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂) (f : RootPairing.Equiv P Q) :
    RootPairing.Equiv Q P where
  weightMap := (weightEquiv P Q f).symm
  coweightMap := (coweightEquiv P Q f).symm
  indexEquiv := f.indexEquiv.symm
  weight_coweight_transpose := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Eq ((↑(RootPairing.Equiv.weightEquiv P Q f).symm).dualMap.comp ↑P.toDualRigh …
    -/
    ext n m
    nth_rw 2 [show m = (weightEquiv P Q f) ((weightEquiv P Q f).symm m) by
      exact (LinearEquiv.symm_apply_eq (weightEquiv P Q f)).mp rfl]
    nth_rw 1 [show n = (coweightEquiv P Q f) ((coweightEquiv P Q f).symm n) by
      exact (LinearEquiv.symm_apply_eq (coweightEquiv P Q f)).mp rfl]
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      n : N
      m : M₂
      ⊢ Eq ((((↑(RootPairing.Equiv.weightEquiv P Q f).symm).dualMap.comp ↑P.toDualRi …
    -/
    have := f.weight_coweight_transpose
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      n : N
      m : M₂
      this : Eq ((↑f).weightMap.dualMap.comp ↑Q.toDualRight) ((↑P.toDualRight).comp  …
      ⊢ Eq ((((↑(RootPairing.Equiv.weightEquiv P Q f).symm).dualMap.comp ↑P.toDualRi …
    -/
    rw [LinearMap.ext_iff₂] at this
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      n : N
      m : M₂
      this : ∀ (m : N₂) (n : M), Eq ((((↑f).weightMap.dualMap.comp ↑Q.toDualRight) m …
      ⊢ Eq ((((↑(RootPairing.Equiv.weightEquiv P Q f).symm).dualMap.comp ↑P.toDualRi …
    -/
    exact Eq.symm (this ((coweightEquiv P Q f).symm n) ((weightEquiv P Q f).symm m))
    /-
      🎉 no goals
    -/
  root_weightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Eq (Function.comp ⇑↑(RootPairing.Equiv.weightEquiv P Q f).symm ⇑Q.root) (Fun …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      ⊢ Eq (Function.comp (⇑↑(RootPairing.Equiv.weightEquiv P Q f).symm) (⇑Q.root) i …
    -/
    simp only [LinearEquiv.coe_coe, comp_apply]
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q f).symm (Q.root i)) (P.root ((↑f).ind …
    -/
    have := f.root_weightMap
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      this : Eq (Function.comp ⇑(↑f).weightMap ⇑P.root) (Function.comp ⇑Q.root ⇑(↑f) …
      ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q f).symm (Q.root i)) (P.root ((↑f).ind …
    -/
    rw [funext_iff] at this
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      this : ∀ (x : ι), Eq (Function.comp (⇑(↑f).weightMap) (⇑P.root) x) (Function.c …
      ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q f).symm (Q.root i)) (P.root ((↑f).ind …
    -/
    specialize this (f.indexEquiv.symm i)
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      this : Eq (Function.comp (⇑(↑f).weightMap) (⇑P.root) ((↑f).indexEquiv.symm i)) …
      ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q f).symm (Q.root i)) (P.root ((↑f).ind …
    -/
    simp only [comp_apply, Equiv.apply_symm_apply] at this
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι₂
      this : Eq ((↑f).weightMap (P.root ((↑f).indexEquiv.symm i))) (Q.root i)
      ⊢ Eq ((RootPairing.Equiv.weightEquiv P Q f).symm (Q.root i)) (P.root ((↑f).ind …
    -/
    simp [← this]
    /-
      🎉 no goals
    -/
  coroot_coweightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Eq (Function.comp ⇑↑(RootPairing.Equiv.coweightEquiv P Q f).symm ⇑P.coroot)  …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      ⊢ Eq (Function.comp (⇑↑(RootPairing.Equiv.coweightEquiv P Q f).symm) (⇑P.coroo …
    -/
    simp only [LinearEquiv.coe_coe, comp_apply, Equiv.symm_symm]
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q f).symm (P.coroot i)) (Q.coroot ((↑ …
    -/
    have := f.coroot_coweightMap
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      this : Eq (Function.comp ⇑(↑f).coweightMap ⇑Q.coroot) (Function.comp ⇑P.coroot …
      ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q f).symm (P.coroot i)) (Q.coroot ((↑ …
    -/
    rw [funext_iff] at this
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      this : ∀ (x : ι₂), Eq (Function.comp (⇑(↑f).coweightMap) (⇑Q.coroot) x) (Funct …
      ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q f).symm (P.coroot i)) (Q.coroot ((↑ …
    -/
    specialize this (f.indexEquiv i)
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      this : Eq (Function.comp (⇑(↑f).coweightMap) (⇑Q.coroot) ((↑f).indexEquiv i))  …
      ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q f).symm (P.coroot i)) (Q.coroot ((↑ …
    -/
    simp only [comp_apply, Equiv.symm_apply_apply] at this
    /-
      case h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      i : ι
      this : Eq ((↑f).coweightMap (Q.coroot ((↑f).indexEquiv i))) (P.coroot i)
      ⊢ Eq ((RootPairing.Equiv.coweightEquiv P Q f).symm (P.coroot i)) (Q.coroot ((↑ …
    -/
    simp [← this]
    /-
      🎉 no goals
    -/
  bijective_weightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Function.Bijective ⇑{ weightMap := ↑(RootPairing.Equiv.weightEquiv P Q f).sy …
    -/
    simp only [LinearEquiv.coe_coe]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Function.Bijective ⇑(RootPairing.Equiv.weightEquiv P Q f).symm
    -/
    exact LinearEquiv.bijective (weightEquiv P Q f).symm
    /-
      🎉 no goals
    -/
  bijective_coweightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Function.Bijective ⇑{ weightMap := ↑(RootPairing.Equiv.weightEquiv P Q f).sy …
    -/
    simp only [LinearEquiv.coe_coe]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹² : CommRing R
      inst✝¹¹ : AddCommGroup M
      inst✝¹⁰ : Module R M
      inst✝⁹ : AddCommGroup N
      inst✝⁸ : Module R N
      ι₂✝ : Type u_5
      M₂✝ : Type u_6
      N₂✝ : Type u_7
      inst✝⁷ : AddCommGroup M₂✝
      inst✝⁶ : Module R M₂✝
      inst✝⁵ : AddCommGroup N₂✝
      inst✝⁴ : Module R N₂✝
      P✝ : RootPairing ι R M N
      Q✝ : RootPairing ι₂✝ R M₂✝ N₂✝
      ι₂ : Type u_8
      M₂ : Type u_9
      N₂ : Type u_10
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      f : P.Equiv Q
      ⊢ Function.Bijective ⇑(RootPairing.Equiv.coweightEquiv P Q f).symm
    -/
    exact LinearEquiv.bijective (coweightEquiv P Q f).symm
    /-
      🎉 no goals
    -/


@[simp]
lemma inv_weightMap {ι₂ M₂ N₂ : Type*} [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂]
    [Module R N₂] (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂)
    (f : RootPairing.Equiv P Q) : (symm P Q f).weightMap = (weightEquiv P Q f).symm :=
  rfl


@[simp]
lemma inv_coweightMap {ι₂ M₂ N₂ : Type*} [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂]
    [Module R N₂] (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂)
    (f : RootPairing.Equiv P Q) : (symm P Q f).coweightMap = (coweightEquiv P Q f).symm :=
  rfl


@[simp]
lemma inv_indexEquiv {ι₂ M₂ N₂ : Type*} [AddCommGroup M₂] [Module R M₂] [AddCommGroup N₂]
    [Module R N₂] (P : RootPairing ι R M N) (Q : RootPairing ι₂ R M₂ N₂)
    (f : RootPairing.Equiv P Q) : (symm P Q f).indexEquiv = (Hom.indexEquiv f.toHom).symm :=
  rfl


/-- Equivalences form a group. -/
instance (P : RootPairing ι R M N) : Group (RootPairing.Equiv P P) where
  mul := comp
  mul_assoc := comp_assoc
  one := id P
  one_mul := id_comp P P
  mul_one := comp_id P P
  inv := symm P P
  inv_mul_cancel e := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      e : P.Equiv P
      ⊢ Eq (HMul.hMul (Inv.inv e) e) 1
    -/
    ext m
      /-
        case weightMap.h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        e : P.Equiv P
        m : M
        ⊢ Eq ((↑(HMul.hMul (Inv.inv e) e)).weightMap m) ((↑1).weightMap m)
      -/
    · rw [← weightEquiv_apply]
      /-
        case weightMap.h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        e : P.Equiv P
        m : M
        ⊢ Eq ((RootPairing.Equiv.weightEquiv P P (HMul.hMul (Inv.inv e) e)) m) ((↑1).w …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case coweightMap.h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        e : P.Equiv P
        m : N
        ⊢ Eq ((↑(HMul.hMul (Inv.inv e) e)).coweightMap m) ((↑1).coweightMap m)
      -/
    · rw [← coweightEquiv_apply]
      /-
        case coweightMap.h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        e : P.Equiv P
        m : N
        ⊢ Eq ((RootPairing.Equiv.coweightEquiv P P (HMul.hMul (Inv.inv e) e)) m) ((↑1) …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case indexEquiv.H
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        e : P.Equiv P
        m : ι
        ⊢ Eq ((↑(HMul.hMul (Inv.inv e) e)).indexEquiv m) ((↑1).indexEquiv m)
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- The automorphism group of a root pairing. -/
abbrev Aut (P : RootPairing ι R M N) := (RootPairing.Equiv P P)


/-- The isomorphism between the automorphism group of a root pairing and the group of invertible
endomorphisms. -/
def toEndUnit (P : RootPairing ι R M N) : Aut P ≃* (End P)ˣ where
  toFun f :=
  { val :=  f.toHom
    inv := (Equiv.symm P P f).toHom
                  /-
                    ι : Type u_1
                    R : Type u_2
                    M : Type u_3
                    N : Type u_4
                    inst✝⁸ : CommRing R
                    inst✝⁷ : AddCommGroup M
                    inst✝⁶ : Module R M
                    inst✝⁵ : AddCommGroup N
                    inst✝⁴ : Module R N
                    ι₂ : Type u_5
                    M₂ : Type u_6
                    N₂ : Type u_7
                    inst✝³ : AddCommGroup M₂
                    inst✝² : Module R M₂
                    inst✝¹ : AddCommGroup N₂
                    inst✝ : Module R N₂
                    P✝ : RootPairing ι R M N
                    Q : RootPairing ι₂ R M₂ N₂
                    P : RootPairing ι R M N
                    f : P.Aut
                    ⊢ Eq (HMul.hMul ↑f ↑(RootPairing.Equiv.symm P P f)) 1
                  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
    val_inv := by ext <;> simp
                          /-
                            🎉 no goals
                          -/
                  /-
                    ι : Type u_1
                    R : Type u_2
                    M : Type u_3
                    N : Type u_4
                    inst✝⁸ : CommRing R
                    inst✝⁷ : AddCommGroup M
                    inst✝⁶ : Module R M
                    inst✝⁵ : AddCommGroup N
                    inst✝⁴ : Module R N
                    ι₂ : Type u_5
                    M₂ : Type u_6
                    N₂ : Type u_7
                    inst✝³ : AddCommGroup M₂
                    inst✝² : Module R M₂
                    inst✝¹ : AddCommGroup N₂
                    inst✝ : Module R N₂
                    P✝ : RootPairing ι R M N
                    Q : RootPairing ι₂ R M₂ N₂
                    P : RootPairing ι R M N
                    f : P.Aut
                    ⊢ Eq (HMul.hMul ↑(RootPairing.Equiv.symm P P f) ↑f) 1
                  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
    inv_val := by ext <;> simp }
                          /-
                            🎉 no goals
                          -/
  invFun f :=
  { f.val with
    bijective_weightMap := by
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ Function.Bijective ⇑__src✝.weightMap
      -/
      refine bijective_iff_has_inverse.mpr ?_
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ Exists fun g => And (Function.LeftInverse g ⇑__src✝.weightMap) (Function.Rig …
      -/
      use f.inv.weightMap
      /-
        case h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ And (Function.LeftInverse ⇑f.inv.weightMap ⇑__src✝.weightMap) (Function.Righ …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Function.LeftInverse ⇑f.inv.weightMap ⇑__src✝.weightMap
        -/
      · refine leftInverse_iff_comp.mpr ?_
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (Function.comp ⇑f.inv.weightMap ⇑__src✝.weightMap) _root_.id
        -/
        simp only [← @LinearMap.coe_comp]
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (⇑(f.inv.weightMap.comp (↑f).weightMap)) _root_.id
        -/
        rw [← Hom.weightMap_mul, f.inv_val, Hom.weightMap_one, LinearMap.id_coe]
        /-
          🎉 no goals
        -/
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Function.RightInverse ⇑f.inv.weightMap ⇑__src✝.weightMap
        -/
      · refine rightInverse_iff_comp.mpr ?_
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (Function.comp ⇑__src✝.weightMap ⇑f.inv.weightMap) _root_.id
        -/
        simp only [← @LinearMap.coe_comp]
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (⇑((↑f).weightMap.comp f.inv.weightMap)) _root_.id
        -/
        rw [← Hom.weightMap_mul, f.val_inv, Hom.weightMap_one, LinearMap.id_coe]
        /-
          🎉 no goals
        -/
    bijective_coweightMap := by
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ Function.Bijective ⇑__src✝.coweightMap
      -/
      refine bijective_iff_has_inverse.mpr ?_
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ Exists fun g => And (Function.LeftInverse g ⇑__src✝.coweightMap) (Function.R …
      -/
      use f.inv.coweightMap
      /-
        case h
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        ι₂ : Type u_5
        M₂ : Type u_6
        N₂ : Type u_7
        inst✝³ : AddCommGroup M₂
        inst✝² : Module R M₂
        inst✝¹ : AddCommGroup N₂
        inst✝ : Module R N₂
        P✝ : RootPairing ι R M N
        Q : RootPairing ι₂ R M₂ N₂
        P : RootPairing ι R M N
        f : Units P.End
        ⊢ And (Function.LeftInverse ⇑f.inv.coweightMap ⇑__src✝.coweightMap) (Function. …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Function.LeftInverse ⇑f.inv.coweightMap ⇑__src✝.coweightMap
        -/
      · refine leftInverse_iff_comp.mpr ?_
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (Function.comp ⇑f.inv.coweightMap ⇑__src✝.coweightMap) _root_.id
        -/
        simp only [← @LinearMap.coe_comp]
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (⇑(f.inv.coweightMap.comp (↑f).coweightMap)) _root_.id
        -/
        rw [← Hom.coweightMap_mul, f.val_inv, Hom.coweightMap_one, LinearMap.id_coe]
        /-
          🎉 no goals
        -/
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Function.RightInverse ⇑f.inv.coweightMap ⇑__src✝.coweightMap
        -/
      · refine rightInverse_iff_comp.mpr ?_
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (Function.comp ⇑__src✝.coweightMap ⇑f.inv.coweightMap) _root_.id
        -/
        simp only [← @LinearMap.coe_comp]
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          M : Type u_3
          N : Type u_4
          inst✝⁸ : CommRing R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          ι₂ : Type u_5
          M₂ : Type u_6
          N₂ : Type u_7
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R M₂
          inst✝¹ : AddCommGroup N₂
          inst✝ : Module R N₂
          P✝ : RootPairing ι R M N
          Q : RootPairing ι₂ R M₂ N₂
          P : RootPairing ι R M N
          f : Units P.End
          ⊢ Eq (⇑((↑f).coweightMap.comp f.inv.coweightMap)) _root_.id
        -/
        rw [← Hom.coweightMap_mul, f.inv_val, Hom.coweightMap_one, LinearMap.id_coe] }
        /-
          🎉 no goals
        -/
                   /-
                     ι : Type u_1
                     R : Type u_2
                     M : Type u_3
                     N : Type u_4
                     inst✝⁸ : CommRing R
                     inst✝⁷ : AddCommGroup M
                     inst✝⁶ : Module R M
                     inst✝⁵ : AddCommGroup N
                     inst✝⁴ : Module R N
                     ι₂ : Type u_5
                     M₂ : Type u_6
                     N₂ : Type u_7
                     inst✝³ : AddCommGroup M₂
                     inst✝² : Module R M₂
                     inst✝¹ : AddCommGroup N₂
                     inst✝ : Module R N₂
                     P✝ : RootPairing ι R M N
                     Q : RootPairing ι₂ R M₂ N₂
                     P : RootPairing ι R M N
                     f : P.Aut
                     ⊢ Eq
                         ((fun f =>
                             let __src := ↑f;
                             { toHom := __src, bijective_weightMap := ⋯, bijective_coweightMap := ⋯ …
                           ((fun f => { val := ↑f, inv := ↑(RootPairing.Equiv.symm P P f), val_inv  …
                         f
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      ι : Type u_1
                      R : Type u_2
                      M : Type u_3
                      N : Type u_4
                      inst✝⁸ : CommRing R
                      inst✝⁷ : AddCommGroup M
                      inst✝⁶ : Module R M
                      inst✝⁵ : AddCommGroup N
                      inst✝⁴ : Module R N
                      ι₂ : Type u_5
                      M₂ : Type u_6
                      N₂ : Type u_7
                      inst✝³ : AddCommGroup M₂
                      inst✝² : Module R M₂
                      inst✝¹ : AddCommGroup N₂
                      inst✝ : Module R N₂
                      P✝ : RootPairing ι R M N
                      Q : RootPairing ι₂ R M₂ N₂
                      P : RootPairing ι R M N
                      f : Units P.End
                      ⊢ Eq
                          ((fun f => { val := ↑f, inv := ↑(RootPairing.Equiv.symm P P f), val_inv := …
                            ((fun f =>
                                let __src := ↑f;
                                { toHom := __src, bijective_weightMap := ⋯, bijective_coweightMap := …
                              f))
                          f
                    -/
  right_inv f := by simp
                    /-
                      🎉 no goals
                    -/
  map_mul' f g := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      f g : P.Aut
      ⊢ Eq
          ({ toFun := fun f => { val := ↑f, inv := ↑(RootPairing.Equiv.symm P P f),  …
                invFun := fun f =>
                  let __src := ↑f;
                  { toHom := __src, bijective_weightMap := ⋯, bijective_coweightMap  …
                left_inv := ⋯, right_inv := ⋯ }.toFun
            (HMul.hMul f g))
          (HMul.hMul
            ({ toFun := fun f => { val := ↑f, inv := ↑(RootPairing.Equiv.symm P P f) …
                  invFun := fun f =>
                    let __src := ↑f;
                    { toHom := __src, bijective_weightMap := ⋯, bijective_coweightMa …
                  left_inv := ⋯, right_inv := ⋯ }.toFun
              f)
            ({ toFun := fun f => { val := ↑f, inv := ↑(RootPairing.Equiv.symm P P f) …
                  invFun := fun f =>
                    let __src := ↑f;
                    { toHom := __src, bijective_weightMap := ⋯, bijective_coweightMa …
                  left_inv := ⋯, right_inv := ⋯ }.toFun
              g))
    -/
    simp only [Equiv.mul_eq_comp, Equiv.toHom_comp]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      f g : P.Aut
      ⊢ Eq { val := (↑f).comp ↑g, inv := ↑(RootPairing.Equiv.symm P P (RootPairing.E …
    -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
    ext <;> simp
            /-
              🎉 no goals
            -/


lemma toEndUnit_val (P : RootPairing ι R M N) (g : Aut P) : (toEndUnit P g).val = g.toHom :=
  rfl


lemma toEndUnit_inv (P : RootPairing ι R M N) (g : Aut P) :
    (toEndUnit P g).inv = (symm P P g).toHom :=
  rfl


/-- The weight space representation of automorphisms -/
@[simps]
def weightHom (P : RootPairing ι R M N) : Aut P →* (M ≃ₗ[R] M) where
  toFun := weightEquiv P P
                 /-
                   ι : Type u_1
                   R : Type u_2
                   M : Type u_3
                   N : Type u_4
                   inst✝⁸ : CommRing R
                   inst✝⁷ : AddCommGroup M
                   inst✝⁶ : Module R M
                   inst✝⁵ : AddCommGroup N
                   inst✝⁴ : Module R N
                   ι₂ : Type u_5
                   M₂ : Type u_6
                   N₂ : Type u_7
                   inst✝³ : AddCommGroup M₂
                   inst✝² : Module R M₂
                   inst✝¹ : AddCommGroup N₂
                   inst✝ : Module R N₂
                   P✝ : RootPairing ι R M N
                   Q : RootPairing ι₂ R M₂ N₂
                   P : RootPairing ι R M N
                   ⊢ Eq (RootPairing.Equiv.weightEquiv P P 1) 1
                 -/
  map_one' := by ext; simp
                      /-
                        🎉 no goals
                      -/
                     /-
                       ι : Type u_1
                       R : Type u_2
                       M : Type u_3
                       N : Type u_4
                       inst✝⁸ : CommRing R
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : AddCommGroup N
                       inst✝⁴ : Module R N
                       ι₂ : Type u_5
                       M₂ : Type u_6
                       N₂ : Type u_7
                       inst✝³ : AddCommGroup M₂
                       inst✝² : Module R M₂
                       inst✝¹ : AddCommGroup N₂
                       inst✝ : Module R N₂
                       P✝ : RootPairing ι R M N
                       Q : RootPairing ι₂ R M₂ N₂
                       P : RootPairing ι R M N
                       x y : P.Aut
                       ⊢ Eq ({ toFun := RootPairing.Equiv.weightEquiv P P, map_one' := ⋯ }.toFun (HMu …
                     -/
  map_mul' x y := by ext; simp
                          /-
                            🎉 no goals
                          -/


lemma weightHom_toLinearMap {P : RootPairing ι R M N} (g : Aut P) :
    (weightHom P g).toLinearMap = Hom.weightHom P g.toHom :=
  rfl


lemma weightHom_injective (P : RootPairing ι R M N) : Injective (Equiv.weightHom P) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Function.Injective ⇑(RootPairing.Equiv.weightHom P)
  -/
  refine Injective.of_comp (f := LinearEquiv.toLinearMap) fun g g' hgg' => ?_
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp LinearEquiv.toLinearMap (⇑(RootPairing.Equiv.weightHo …
    ⊢ Eq g g'
  -/
  let h : (weightHom P g).toLinearMap = (weightHom P g').toLinearMap := hgg' --`have` gets lint
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp LinearEquiv.toLinearMap (⇑(RootPairing.Equiv.weightHo …
    h : Eq ↑((RootPairing.Equiv.weightHom P) g) ↑((RootPairing.Equiv.weightHom P)  …
    ⊢ Eq g g'
  -/
  rw [weightHom_toLinearMap, weightHom_toLinearMap] at h
  suffices h' : g.toHom = g'.toHom by
    exact Equiv.ext hgg' (congrArg Hom.coweightMap h') (congrArg Hom.indexEquiv h')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp LinearEquiv.toLinearMap (⇑(RootPairing.Equiv.weightHo …
    h : Eq ((RootPairing.Hom.weightHom P) ↑g) ((RootPairing.Hom.weightHom P) ↑g')
    ⊢ Eq ↑g ↑g'
  -/
  exact Hom.weightHom_injective P hgg'
  /-
    🎉 no goals
  -/


@[simp]
lemma weightEquiv_inv {P : RootPairing ι R M N} (g : Aut P) :
    weightEquiv P P g⁻¹ = (weightEquiv P P g)⁻¹ :=
  LinearEquiv.toLinearMap_inj.mp rfl


/-- The coweight space representation of automorphisms -/
@[simps]
def coweightHom (P : RootPairing ι R M N) : Aut P →* (N ≃ₗ[R] N)ᵐᵒᵖ where
  toFun g := MulOpposite.op (coweightEquiv P P g)
  map_one' := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      ⊢ Eq ((fun g => MulOpposite.op (RootPairing.Equiv.coweightEquiv P P g)) 1) 1
    -/
    simp only [MulOpposite.op_eq_one_iff]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      ⊢ Eq (RootPairing.Equiv.coweightEquiv P P 1) 1
    -/
    exact LinearEquiv.toLinearMap_inj.mp rfl
    /-
      🎉 no goals
    -/
  map_mul' := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      ⊢ ∀ (x y : P.Aut), Eq ({ toFun := fun g => MulOpposite.op (RootPairing.Equiv.c …
    -/
    simp only [mul_eq_comp, coweightEquiv_comp_toLin]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      ⊢ ∀ (x y : P.Aut), Eq (MulOpposite.op ((RootPairing.Equiv.coweightEquiv P P x) …
    -/
    exact fun x y ↦ rfl
    /-
      🎉 no goals
    -/


lemma coweightHom_toLinearMap {P : RootPairing ι R M N} (g : Aut P) :
    (MulOpposite.unop (coweightHom P g)).toLinearMap =
      MulOpposite.unop (Hom.coweightHom P g.toHom) :=
  rfl


lemma coweightHom_injective (P : RootPairing ι R M N) : Injective (Equiv.coweightHom P) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    ⊢ Function.Injective ⇑(RootPairing.Equiv.coweightHom P)
  -/
  refine Injective.of_comp (f := fun a => MulOpposite.op a) fun g g' hgg' => ?_
  have h : (MulOpposite.unop (coweightHom P g)).toLinearMap =
      (MulOpposite.unop (coweightHom P g')).toLinearMap := by
    simp_all
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp (fun a => MulOpposite.op a) (⇑(RootPairing.Equiv.cowe …
    h : Eq ↑(MulOpposite.unop ((RootPairing.Equiv.coweightHom P) g)) ↑(MulOpposite …
    ⊢ Eq g g'
  -/
  rw [coweightHom_toLinearMap, coweightHom_toLinearMap] at h
  suffices h' : g.toHom = g'.toHom by
    exact Equiv.ext (congrArg Hom.weightMap h') h (congrArg Hom.indexEquiv h')
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp (fun a => MulOpposite.op a) (⇑(RootPairing.Equiv.cowe …
    h : Eq (MulOpposite.unop ((RootPairing.Hom.coweightHom P) ↑g)) (MulOpposite.un …
    ⊢ Eq ↑g ↑g'
  -/
  apply Hom.coweightHom_injective P
  /-
    case a
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    g g' : P.Aut
    hgg' : Eq (Function.comp (fun a => MulOpposite.op a) (⇑(RootPairing.Equiv.cowe …
    h : Eq (MulOpposite.unop ((RootPairing.Hom.coweightHom P) ↑g)) (MulOpposite.un …
    ⊢ Eq ((RootPairing.Hom.coweightHom P) ↑g) ((RootPairing.Hom.coweightHom P) ↑g')
  -/
  exact MulOpposite.unop_inj.mp h
  /-
    🎉 no goals
  -/


lemma coweightHom_op {P : RootPairing ι R M N} (g : Aut P) :
    MulOpposite.unop (coweightHom P g) = coweightEquiv P P g :=
  rfl


@[simp]
lemma coweightEquiv_inv {P : RootPairing ι R M N} (g : Aut P) :
    coweightEquiv P P g⁻¹ = (coweightEquiv P P g)⁻¹ :=
  LinearEquiv.toLinearMap_inj.mp rfl


/-- The permutation representation of the automorphism group on the root index set -/
@[simps]
def indexHom (P : RootPairing ι R M N) : Aut P →* (ι ≃ ι) where
  toFun g := g.toHom.indexEquiv
                 /-
                   ι : Type u_1
                   R : Type u_2
                   M : Type u_3
                   N : Type u_4
                   inst✝⁸ : CommRing R
                   inst✝⁷ : AddCommGroup M
                   inst✝⁶ : Module R M
                   inst✝⁵ : AddCommGroup N
                   inst✝⁴ : Module R N
                   ι₂ : Type u_5
                   M₂ : Type u_6
                   N₂ : Type u_7
                   inst✝³ : AddCommGroup M₂
                   inst✝² : Module R M₂
                   inst✝¹ : AddCommGroup N₂
                   inst✝ : Module R N₂
                   P✝ : RootPairing ι R M N
                   Q : RootPairing ι₂ R M₂ N₂
                   P : RootPairing ι R M N
                   ⊢ Eq ((fun g => (↑g).indexEquiv) 1) 1
                 -/
  map_one' := by ext; simp
                      /-
                        🎉 no goals
                      -/
                     /-
                       ι : Type u_1
                       R : Type u_2
                       M : Type u_3
                       N : Type u_4
                       inst✝⁸ : CommRing R
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : AddCommGroup N
                       inst✝⁴ : Module R N
                       ι₂ : Type u_5
                       M₂ : Type u_6
                       N₂ : Type u_7
                       inst✝³ : AddCommGroup M₂
                       inst✝² : Module R M₂
                       inst✝¹ : AddCommGroup N₂
                       inst✝ : Module R N₂
                       P✝ : RootPairing ι R M N
                       Q : RootPairing ι₂ R M₂ N₂
                       P : RootPairing ι R M N
                       x y : P.Aut
                       ⊢ Eq ({ toFun := fun g => (↑g).indexEquiv, map_one' := ⋯ }.toFun (HMul.hMul x  …
                     -/
  map_mul' x y := by ext; simp
                          /-
                            🎉 no goals
                          -/


@[simp]
lemma indexEquiv_inv {P : RootPairing ι R M N} (g : Aut P) :
    (g⁻¹).toHom.indexEquiv = (indexHom P g)⁻¹ :=
  rfl


/-- The automorphism of a root pairing given by a reflection. -/
def reflection (P : RootPairing ι R M N) (i : ι) : Aut P where
  weightMap := P.reflection i
  coweightMap := P.coreflection i
  indexEquiv := P.reflection_perm i
  weight_coweight_transpose := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      ⊢ Eq ((↑(P.reflection i)).dualMap.comp ↑P.toDualRight) ((↑P.toDualRight).comp  …
    -/
    ext f x
    simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, comp_apply,
      PerfectPairing.toDualRight_apply, LinearMap.dualMap_apply, PerfectPairing.flip_apply_apply,
      LinearEquiv.comp_coe, LinearEquiv.trans_apply]
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      f : N
      x : M
      ⊢ Eq ((P.toPerfectPairing ((P.reflection i) x)) f) ((P.toPerfectPairing x) ((P …
    -/
    rw [RootPairing.reflection_apply, RootPairing.coreflection_apply]
    simp only [← PerfectPairing.toLin_apply, map_sub, map_smul, LinearMap.sub_apply,
      toLin_toPerfectPairing, LinearMap.smul_apply, smul_eq_mul, sub_right_inj]
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      f : N
      x : M
      ⊢ Eq (HMul.hMul ((P.flip.toLin (P.coroot i)) x) ((P.toPerfectPairing (P.root i …
    -/
    simp only [PerfectPairing.toLin_apply, PerfectPairing.flip_apply_apply, mul_comm]
    /-
      🎉 no goals
    -/
                       /-
                         ι : Type u_1
                         R : Type u_2
                         M : Type u_3
                         N : Type u_4
                         inst✝⁸ : CommRing R
                         inst✝⁷ : AddCommGroup M
                         inst✝⁶ : Module R M
                         inst✝⁵ : AddCommGroup N
                         inst✝⁴ : Module R N
                         ι₂ : Type u_5
                         M₂ : Type u_6
                         N₂ : Type u_7
                         inst✝³ : AddCommGroup M₂
                         inst✝² : Module R M₂
                         inst✝¹ : AddCommGroup N₂
                         inst✝ : Module R N₂
                         P✝ : RootPairing ι R M N
                         Q : RootPairing ι₂ R M₂ N₂
                         P : RootPairing ι R M N
                         i : ι
                         ⊢ Eq (Function.comp ⇑↑(P.reflection i) ⇑P.root) (Function.comp ⇑P.root ⇑(P.ref …
                       -/
  root_weightMap := by ext; simp
                            /-
                              🎉 no goals
                            -/
                           /-
                             ι : Type u_1
                             R : Type u_2
                             M : Type u_3
                             N : Type u_4
                             inst✝⁸ : CommRing R
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : AddCommGroup N
                             inst✝⁴ : Module R N
                             ι₂ : Type u_5
                             M₂ : Type u_6
                             N₂ : Type u_7
                             inst✝³ : AddCommGroup M₂
                             inst✝² : Module R M₂
                             inst✝¹ : AddCommGroup N₂
                             inst✝ : Module R N₂
                             P✝ : RootPairing ι R M N
                             Q : RootPairing ι₂ R M₂ N₂
                             P : RootPairing ι R M N
                             i : ι
                             ⊢ Eq (Function.comp ⇑↑(P.coreflection i) ⇑P.coroot) (Function.comp ⇑P.coroot ⇑ …
                           -/
  coroot_coweightMap := by ext; simp
                                /-
                                  🎉 no goals
                                -/
  bijective_weightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      ⊢ Function.Bijective ⇑{ weightMap := ↑(P.reflection i), coweightMap := ↑(P.cor …
    -/
    simp only [LinearEquiv.coe_coe]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      ⊢ Function.Bijective ⇑(P.reflection i)
    -/
    exact LinearEquiv.bijective (P.reflection i)
    /-
      🎉 no goals
    -/
  bijective_coweightMap := by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      ⊢ Function.Bijective ⇑{ weightMap := ↑(P.reflection i), coweightMap := ↑(P.cor …
    -/
    simp only [LinearEquiv.coe_coe]
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      ι₂ : Type u_5
      M₂ : Type u_6
      N₂ : Type u_7
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommGroup N₂
      inst✝ : Module R N₂
      P✝ : RootPairing ι R M N
      Q : RootPairing ι₂ R M₂ N₂
      P : RootPairing ι R M N
      i : ι
      ⊢ Function.Bijective ⇑(P.coreflection i)
    -/
    exact LinearEquiv.bijective (P.coreflection i)
    /-
      🎉 no goals
    -/


@[simp]
lemma reflection_weightEquiv (P : RootPairing ι R M N) (i : ι) :
    (reflection P i).weightEquiv = P.reflection i :=
  LinearEquiv.toLinearMap_inj.mp rfl


@[simp]
lemma reflection_coweightEquiv (P : RootPairing ι R M N) (i : ι) :
    (reflection P i).coweightEquiv = P.coreflection i :=
  LinearEquiv.toLinearMap_inj.mp rfl


@[simp]
lemma reflection_indexEquiv (P : RootPairing ι R M N) (i : ι) :
    (reflection P i).indexEquiv = P.reflection_perm i :=
  rfl


@[simp]
lemma reflection_inv (P : RootPairing ι R M N) (i : ι) :
    (reflection P i)⁻¹ = (reflection P i) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    P : RootPairing ι R M N
    i : ι
    ⊢ Eq (Inv.inv (RootPairing.Equiv.reflection P i)) (RootPairing.Equiv.reflectio …
  -/
  refine Equiv.ext ?_ ?_ ?_
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Eq (↑(Inv.inv (RootPairing.Equiv.reflection P i))).weightMap (↑(RootPairing. …
    -/
  · exact LinearMap.ext_iff.mpr (fun x => by simp [← weightEquiv_apply])
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Eq (↑(Inv.inv (RootPairing.Equiv.reflection P i))).coweightMap (↑(RootPairin …
    -/
  · exact LinearMap.ext_iff.mpr (fun x => by simp [← coweightEquiv_apply])
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      P : RootPairing ι R M N
      i : ι
      ⊢ Eq (↑(Inv.inv (RootPairing.Equiv.reflection P i))).indexEquiv (↑(RootPairing …
    -/
  · exact _root_.Equiv.ext (fun j => by simp only [← indexHom_apply, map_inv]; simp)
    /-
      🎉 no goals
    -/


instance : MulAction P.Aut M where
  smul w v := Equiv.weightHom P w v
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


instance : MulAction (P.Aut)ᵐᵒᵖ N where
  smul w v := (MulOpposite.unop (Equiv.coweightHom P (MulOpposite.unop w))) v
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


instance : MulAction P.Aut ι where
  smul w i := Equiv.indexHom P w i
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


