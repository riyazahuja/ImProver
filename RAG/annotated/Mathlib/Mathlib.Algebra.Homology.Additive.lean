instance : Zero (C ⟶ D) :=
  ⟨{ f := fun _ => 0 }⟩


instance : Add (C ⟶ D) :=
  ⟨fun f g => { f := fun i => f.f i + g.f i }⟩


instance : Neg (C ⟶ D) :=
  ⟨fun f => { f := fun i => -f.f i }⟩


instance : Sub (C ⟶ D) :=
  ⟨fun f g => { f := fun i => f.f i - g.f i }⟩


instance hasNatScalar : SMul ℕ (C ⟶ D) :=
  ⟨fun n f =>
    { f := fun i => n • f.f i
                               /-
                                 ι : Type u_1
                                 V : Type u
                                 inst✝⁷ : CategoryTheory.Category.{v, u} V
                                 inst✝⁶ : CategoryTheory.Preadditive V
                                 W : Type u_2
                                 inst✝⁵ : CategoryTheory.Category.{?u.4661, u_2} W
                                 inst✝⁴ : CategoryTheory.Preadditive W
                                 W₁ : Type u_3
                                 W₂ : Type u_4
                                 inst✝³ : CategoryTheory.Category.{?u.4687, u_3} W₁
                                 inst✝² : CategoryTheory.Category.{?u.4691, u_4} W₂
                                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                 c : ComplexShape ι
                                 C D : HomologicalComplex V c
                                 f✝ : Quiver.Hom C D
                                 i✝ : ι
                                 n : Nat
                                 f : Quiver.Hom C D
                                 i j : ι
                                 x✝ : c.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => HSMul.hSMul n (f.f i)) i)  …
                               -/
      comm' := fun i j _ => by simp [Preadditive.nsmul_comp, Preadditive.comp_nsmul] }⟩
                               /-
                                 🎉 no goals
                               -/


instance hasIntScalar : SMul ℤ (C ⟶ D) :=
  ⟨fun n f =>
    { f := fun i => n • f.f i
                               /-
                                 ι : Type u_1
                                 V : Type u
                                 inst✝⁷ : CategoryTheory.Category.{v, u} V
                                 inst✝⁶ : CategoryTheory.Preadditive V
                                 W : Type u_2
                                 inst✝⁵ : CategoryTheory.Category.{?u.5978, u_2} W
                                 inst✝⁴ : CategoryTheory.Preadditive W
                                 W₁ : Type u_3
                                 W₂ : Type u_4
                                 inst✝³ : CategoryTheory.Category.{?u.6004, u_3} W₁
                                 inst✝² : CategoryTheory.Category.{?u.6008, u_4} W₂
                                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                 c : ComplexShape ι
                                 C D : HomologicalComplex V c
                                 f✝ : Quiver.Hom C D
                                 i✝ : ι
                                 n : Int
                                 f : Quiver.Hom C D
                                 i j : ι
                                 x✝ : c.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => HSMul.hSMul n (f.f i)) i)  …
                               -/
      comm' := fun i j _ => by simp [Preadditive.zsmul_comp, Preadditive.comp_zsmul] }⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem zero_f_apply (i : ι) : (0 : C ⟶ D).f i = 0 :=
  rfl


@[simp]
theorem add_f_apply (f g : C ⟶ D) (i : ι) : (f + g).f i = f.f i + g.f i :=
  rfl


@[simp]
theorem neg_f_apply (f : C ⟶ D) (i : ι) : (-f).f i = -f.f i :=
  rfl


@[simp]
theorem sub_f_apply (f g : C ⟶ D) (i : ι) : (f - g).f i = f.f i - g.f i :=
  rfl


@[simp]
theorem nsmul_f_apply (n : ℕ) (f : C ⟶ D) (i : ι) : (n • f).f i = n • f.f i :=
  rfl


@[simp]
theorem zsmul_f_apply (n : ℤ) (f : C ⟶ D) (i : ι) : (n • f).f i = n • f.f i :=
  rfl


instance : AddCommGroup (C ⟶ D) :=
  Function.Injective.addCommGroup Hom.f HomologicalComplex.hom_f_injective
        /-
          ι : Type u_1
          V : Type u
          inst✝⁷ : CategoryTheory.Category.{v, u} V
          inst✝⁶ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁵ : CategoryTheory.Category.{?u.11046, u_2} W
          inst✝⁴ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝³ : CategoryTheory.Category.{?u.11072, u_3} W₁
          inst✝² : CategoryTheory.Category.{?u.11076, u_4} W₂
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c : ComplexShape ι
          C D : HomologicalComplex V c
          f : Quiver.Hom C D
          i : ι
          ⊢ Eq (HomologicalComplex.Hom.f 0) 0
        -/
        /-
          🎉 no goals
        -/
                       /-
                         🎉 no goals
                       -/
                                      /-
                                        🎉 no goals
                                      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    (by aesop_cat) (by aesop_cat) (by aesop_cat) (by aesop_cat) (by aesop_cat) (by aesop_cat)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/

-- Porting note: proofs had to be provided here, otherwise Lean tries to apply
-- `Preadditive.add_comp/comp_add` to `HomologicalComplex V c`

instance : Preadditive (HomologicalComplex V c) where
  add_comp _ _ _ f f' g := by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f f' : Quiver.Hom x✝² x✝¹
      g : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f f') g) (HAdd.hAdd (Categ …
    -/
    ext
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f f' : Quiver.Hom x✝² x✝¹
      g : Quiver.Hom x✝¹ x✝
      i✝ : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (HAdd.hAdd f f') g).f i✝) ((HAdd.hAd …
    -/
    simp only [comp_f, add_f_apply]
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f f' : Quiver.Hom x✝² x✝¹
      g : Quiver.Hom x✝¹ x✝
      i✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (f.f i✝) (f'.f i✝)) (g.f i …
    -/
    rw [Preadditive.add_comp]
    /-
      🎉 no goals
    -/
  comp_add _ _ _ f g g' := by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f : Quiver.Hom x✝² x✝¹
      g g' : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd g g')) (HAdd.hAdd (Categ …
    -/
    ext
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f : Quiver.Hom x✝² x✝¹
      g g' : Quiver.Hom x✝¹ x✝
      i✝ : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (HAdd.hAdd g g')).f i✝) ((HAdd.hAd …
    -/
    simp only [comp_f, add_f_apply]
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝⁷ : CategoryTheory.Category.{v, u} V
      inst✝⁶ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.14287, u_2} W
      inst✝⁴ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝³ : CategoryTheory.Category.{?u.14313, u_3} W₁
      inst✝² : CategoryTheory.Category.{?u.14317, u_4} W₂
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f✝ : Quiver.Hom C D
      i : ι
      x✝² x✝¹ x✝ : HomologicalComplex V c
      f : Quiver.Hom x✝² x✝¹
      g g' : Quiver.Hom x✝¹ x✝
      i✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i✝) (HAdd.hAdd (g.f i✝) (g'.f i✝ …
    -/
    rw [Preadditive.comp_add]
    /-
      🎉 no goals
    -/


/-- The `i`-th component of a chain map, as an additive map from chain maps to morphisms. -/
@[simps!]
def Hom.fAddMonoidHom {C₁ C₂ : HomologicalComplex V c} (i : ι) : (C₁ ⟶ C₂) →+ (C₁.X i ⟶ C₂.X i) :=
  AddMonoidHom.mk' (fun f => Hom.f f i) fun _ _ => rfl


instance eval_additive (i : ι) : (eval V c i).Additive where


/-- An additive functor induces a functor between homological complexes.
This is sometimes called the "prolongation".
-/
@[simps]
def Functor.mapHomologicalComplex (F : W₁ ⥤ W₂) [F.PreservesZeroMorphisms] (c : ComplexShape ι) :
    HomologicalComplex W₁ c ⥤ HomologicalComplex W₂ c where
  obj C :=
    { X := fun i => F.obj (C.X i)
      d := fun i j => F.map (C.d i j)
      shape := fun i j w => by
        /-
          ι : Type u_1
          V : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} V
          inst✝⁷ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁶ : CategoryTheory.Category.{?u.18927, u_2} W
          inst✝⁵ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁴ : CategoryTheory.Category.{?u.18953, u_3} W₁
          inst✝³ : CategoryTheory.Category.{?u.18957, u_4} W₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C✝ D : HomologicalComplex V c✝
          f : Quiver.Hom C✝ D
          i✝ : ι
          F : CategoryTheory.Functor W₁ W₂
          inst✝ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          C : HomologicalComplex W₁ c
          i j : ι
          w : Not (c.Rel i j)
          ⊢ Eq ((fun i j => F.map (C.d i j)) i j) 0
        -/
        dsimp only
        /-
          ι : Type u_1
          V : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} V
          inst✝⁷ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁶ : CategoryTheory.Category.{?u.18927, u_2} W
          inst✝⁵ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁴ : CategoryTheory.Category.{?u.18953, u_3} W₁
          inst✝³ : CategoryTheory.Category.{?u.18957, u_4} W₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C✝ D : HomologicalComplex V c✝
          f : Quiver.Hom C✝ D
          i✝ : ι
          F : CategoryTheory.Functor W₁ W₂
          inst✝ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          C : HomologicalComplex W₁ c
          i j : ι
          w : Not (c.Rel i j)
          ⊢ Eq (F.map (C.d i j)) 0
        -/
        rw [C.shape _ _ w, F.map_zero]
        /-
          🎉 no goals
        -/
                                       /-
                                         ι : Type u_1
                                         V : Type u
                                         inst✝⁸ : CategoryTheory.Category.{v, u} V
                                         inst✝⁷ : CategoryTheory.Preadditive V
                                         W : Type u_2
                                         inst✝⁶ : CategoryTheory.Category.{?u.18927, u_2} W
                                         inst✝⁵ : CategoryTheory.Preadditive W
                                         W₁ : Type u_3
                                         W₂ : Type u_4
                                         inst✝⁴ : CategoryTheory.Category.{?u.18953, u_3} W₁
                                         inst✝³ : CategoryTheory.Category.{?u.18957, u_4} W₂
                                         inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
                                         inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                         c✝ : ComplexShape ι
                                         C✝ D : HomologicalComplex V c✝
                                         f : Quiver.Hom C✝ D
                                         i✝ : ι
                                         F : CategoryTheory.Functor W₁ W₂
                                         inst✝ : F.PreservesZeroMorphisms
                                         c : ComplexShape ι
                                         C : HomologicalComplex W₁ c
                                         i j k : ι
                                         x✝¹ : c.Rel i j
                                         x✝ : c.Rel j k
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => F.map (C.d i j)) i j) (( …
                                       -/
      d_comp_d' := fun i j k _ _ => by rw [← F.map_comp, C.d_comp_d, F.map_zero] }
                                       /-
                                         🎉 no goals
                                       -/
  map f :=
    { f := fun i => F.map (f.f i)
      comm' := fun i j _ => by
        /-
          ι : Type u_1
          V : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} V
          inst✝⁷ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁶ : CategoryTheory.Category.{?u.18927, u_2} W
          inst✝⁵ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁴ : CategoryTheory.Category.{?u.18953, u_3} W₁
          inst✝³ : CategoryTheory.Category.{?u.18957, u_4} W₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C D : HomologicalComplex V c✝
          f✝ : Quiver.Hom C D
          i✝ : ι
          F : CategoryTheory.Functor W₁ W₂
          inst✝ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          X✝ Y✝ : HomologicalComplex W₁ c
          f : Quiver.Hom X✝ Y✝
          i j : ι
          x✝ : c.Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => F.map (f.f i)) i) (((fun C …
        -/
        dsimp
        /-
          ι : Type u_1
          V : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} V
          inst✝⁷ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁶ : CategoryTheory.Category.{?u.18927, u_2} W
          inst✝⁵ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁴ : CategoryTheory.Category.{?u.18953, u_3} W₁
          inst✝³ : CategoryTheory.Category.{?u.18957, u_4} W₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C D : HomologicalComplex V c✝
          f✝ : Quiver.Hom C D
          i✝ : ι
          F : CategoryTheory.Functor W₁ W₂
          inst✝ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          X✝ Y✝ : HomologicalComplex W₁ c
          f : Quiver.Hom X✝ Y✝
          i j : ι
          x✝ : c.Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (f.f i)) (F.map (Y✝.d i j))) ( …
        -/
        rw [← F.map_comp, ← F.map_comp, f.comm] }
        /-
          🎉 no goals
        -/


instance (F : W₁ ⥤ W₂) [F.PreservesZeroMorphisms] (c : ComplexShape ι) :
    (F.mapHomologicalComplex c).PreservesZeroMorphisms where


instance Functor.map_homogical_complex_additive (F : V ⥤ W) [F.Additive] (c : ComplexShape ι) :
    (F.mapHomologicalComplex c).Additive where


/-- The functor on homological complexes induced by the identity functor is
isomorphic to the identity functor. -/
@[simps!]
def Functor.mapHomologicalComplexIdIso (c : ComplexShape ι) :
    (𝟭 W₁).mapHomologicalComplex c ≅ 𝟭 _ :=
                               /-
                                 ι : Type u_1
                                 V : Type u
                                 inst✝⁷ : CategoryTheory.Category.{v, u} V
                                 inst✝⁶ : CategoryTheory.Preadditive V
                                 W : Type u_2
                                 inst✝⁵ : CategoryTheory.Category.{?u.37298, u_2} W
                                 inst✝⁴ : CategoryTheory.Preadditive W
                                 W₁ : Type u_3
                                 W₂ : Type u_4
                                 inst✝³ : CategoryTheory.Category.{?u.37324, u_3} W₁
                                 inst✝² : CategoryTheory.Category.{?u.37328, u_4} W₂
                                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₁
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                 c✝ : ComplexShape ι
                                 C D : HomologicalComplex V c✝
                                 f : Quiver.Hom C D
                                 i : ι
                                 c : ComplexShape ι
                                 K : HomologicalComplex W₁ c
                                 ⊢ ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun x => C …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun K => Hom.isoOfComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


instance Functor.mapHomologicalComplex_reflects_iso (F : W₁ ⥤ W₂) [F.PreservesZeroMorphisms]
    [ReflectsIsomorphisms F] (c : ComplexShape ι) :
    ReflectsIsomorphisms (F.mapHomologicalComplex c) :=
  ⟨fun f => by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁹ : CategoryTheory.Category.{v, u} V
      inst✝⁸ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁷ : CategoryTheory.Category.{?u.43652, u_2} W
      inst✝⁶ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝⁵ : CategoryTheory.Category.{u_5, u_3} W₁
      inst✝⁴ : CategoryTheory.Category.{u_6, u_4} W₂
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₂
      c✝ : ComplexShape ι
      C D : HomologicalComplex V c✝
      f✝ : Quiver.Hom C D
      i : ι
      F : CategoryTheory.Functor W₁ W₂
      inst✝¹ : F.PreservesZeroMorphisms
      inst✝ : F.ReflectsIsomorphisms
      c : ComplexShape ι
      A✝ B✝ : HomologicalComplex W₁ c
      f : Quiver.Hom A✝ B✝
      ⊢ ∀ [inst : CategoryTheory.IsIso ((F.mapHomologicalComplex c).map f)], Categor …
    -/
    intro
    haveI : ∀ n : ι, IsIso (F.map (f.f n)) := fun n =>
        ((HomologicalComplex.eval W₂ c n).mapIso
          (asIso ((F.mapHomologicalComplex c).map f))).isIso_hom
    /-
      ι : Type u_1
      V : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} V
      inst✝⁹ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁸ : CategoryTheory.Category.{?u.43652, u_2} W
      inst✝⁷ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_3} W₁
      inst✝⁵ : CategoryTheory.Category.{u_6, u_4} W₂
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c✝ : ComplexShape ι
      C D : HomologicalComplex V c✝
      f✝ : Quiver.Hom C D
      i : ι
      F : CategoryTheory.Functor W₁ W₂
      inst✝² : F.PreservesZeroMorphisms
      inst✝¹ : F.ReflectsIsomorphisms
      c : ComplexShape ι
      A✝ B✝ : HomologicalComplex W₁ c
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso ((F.mapHomologicalComplex c).map f)
      this : ∀ (n : ι), CategoryTheory.IsIso (F.map (f.f n))
      ⊢ CategoryTheory.IsIso f
    -/
    haveI := fun n => isIso_of_reflects_iso (f.f n) F
    /-
      ι : Type u_1
      V : Type u
      inst✝¹⁰ : CategoryTheory.Category.{v, u} V
      inst✝⁹ : CategoryTheory.Preadditive V
      W : Type u_2
      inst✝⁸ : CategoryTheory.Category.{?u.43652, u_2} W
      inst✝⁷ : CategoryTheory.Preadditive W
      W₁ : Type u_3
      W₂ : Type u_4
      inst✝⁶ : CategoryTheory.Category.{u_5, u_3} W₁
      inst✝⁵ : CategoryTheory.Category.{u_6, u_4} W₂
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₁
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms W₂
      c✝ : ComplexShape ι
      C D : HomologicalComplex V c✝
      f✝ : Quiver.Hom C D
      i : ι
      F : CategoryTheory.Functor W₁ W₂
      inst✝² : F.PreservesZeroMorphisms
      inst✝¹ : F.ReflectsIsomorphisms
      c : ComplexShape ι
      A✝ B✝ : HomologicalComplex W₁ c
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso ((F.mapHomologicalComplex c).map f)
      this✝ : ∀ (n : ι), CategoryTheory.IsIso (F.map (f.f n))
      this : ∀ (n : ι), CategoryTheory.IsIso (f.f n)
      ⊢ CategoryTheory.IsIso f
    -/
    exact HomologicalComplex.Hom.isIso_of_components f⟩
    /-
      🎉 no goals
    -/


/-- A natural transformation between functors induces a natural transformation
between those functors applied to homological complexes.
-/
@[simps]
def NatTrans.mapHomologicalComplex {F G : W₁ ⥤ W₂}
    [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms] (α : F ⟶ G)
    (c : ComplexShape ι) : F.mapHomologicalComplex c ⟶ G.mapHomologicalComplex c where
  app C := { f := fun _ => α.app _ }


@[simp]
theorem NatTrans.mapHomologicalComplex_id
    (c : ComplexShape ι) (F : W₁ ⥤ W₂) [F.PreservesZeroMorphisms] :
                                                                                 /-
                                                                                   ι : Type u_1
                                                                                   W₁ : Type u_3
                                                                                   W₂ : Type u_4
                                                                                   inst✝⁴ : CategoryTheory.Category.{u_5, u_3} W₁
                                                                                   inst✝³ : CategoryTheory.Category.{u_6, u_4} W₂
                                                                                   inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₁
                                                                                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                                                                   c : ComplexShape ι
                                                                                   F : CategoryTheory.Functor W₁ W₂
                                                                                   inst✝ : F.PreservesZeroMorphisms
                                                                                   ⊢ Eq (CategoryTheory.NatTrans.mapHomologicalComplex (CategoryTheory.CategorySt …
                                                                                 -/
    NatTrans.mapHomologicalComplex (𝟙 F) c = 𝟙 (F.mapHomologicalComplex c) := by aesop_cat
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem NatTrans.mapHomologicalComplex_comp (c : ComplexShape ι) {F G H : W₁ ⥤ W₂}
    [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms] [H.PreservesZeroMorphisms]
    (α : F ⟶ G) (β : G ⟶ H) :
    NatTrans.mapHomologicalComplex (α ≫ β) c =
      NatTrans.mapHomologicalComplex α c ≫ NatTrans.mapHomologicalComplex β c := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} W₁
    inst✝⁵ : CategoryTheory.Category.{u_6, u_4} W₂
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms W₂
    c : ComplexShape ι
    F G H : CategoryTheory.Functor W₁ W₂
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : G.PreservesZeroMorphisms
    inst✝ : H.PreservesZeroMorphisms
    α : Quiver.Hom F G
    β : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.NatTrans.mapHomologicalComplex (CategoryTheory.CategorySt …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp 1100)]
theorem NatTrans.mapHomologicalComplex_naturality {c : ComplexShape ι} {F G : W₁ ⥤ W₂}
    [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms]
    (α : F ⟶ G) {C D : HomologicalComplex W₁ c} (f : C ⟶ D) :
    (F.mapHomologicalComplex c).map f ≫ (NatTrans.mapHomologicalComplex α c).app D =
      (NatTrans.mapHomologicalComplex α c).app C ≫ (G.mapHomologicalComplex c).map f := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_5, u_3} W₁
    inst✝⁴ : CategoryTheory.Category.{u_6, u_4} W₂
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms W₂
    c : ComplexShape ι
    F G : CategoryTheory.Functor W₁ W₂
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : G.PreservesZeroMorphisms
    α : Quiver.Hom F G
    C D : HomologicalComplex W₁ c
    f : Quiver.Hom C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapHomologicalComplex c).map f) ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- A natural isomorphism between functors induces a natural isomorphism
between those functors applied to homological complexes.
-/
@[simps!]
def NatIso.mapHomologicalComplex {F G : W₁ ⥤ W₂} [F.PreservesZeroMorphisms]
    [G.PreservesZeroMorphisms] (α : F ≅ G) (c : ComplexShape ι) :
    F.mapHomologicalComplex c ≅ G.mapHomologicalComplex c where
  hom := NatTrans.mapHomologicalComplex α.hom c
  inv := NatTrans.mapHomologicalComplex α.inv c
  hom_inv_id := by simp only [← NatTrans.mapHomologicalComplex_comp, α.hom_inv_id,
    NatTrans.mapHomologicalComplex_id]
  inv_hom_id := by simp only [← NatTrans.mapHomologicalComplex_comp, α.inv_hom_id,
    NatTrans.mapHomologicalComplex_id]


/-- An equivalence of categories induces an equivalences between the respective categories
of homological complex.
-/
@[simps]
def Equivalence.mapHomologicalComplex (e : W₁ ≌ W₂) [e.functor.PreservesZeroMorphisms]
    (c : ComplexShape ι) :
    HomologicalComplex W₁ c ≌ HomologicalComplex W₂ c where
  functor := e.functor.mapHomologicalComplex c
  inverse := e.inverse.mapHomologicalComplex c
  unitIso :=
    (Functor.mapHomologicalComplexIdIso W₁ c).symm ≪≫ NatIso.mapHomologicalComplex e.unitIso c
  counitIso := NatIso.mapHomologicalComplex e.counitIso c ≪≫
  Functor.mapHomologicalComplexIdIso W₂ c


theorem map_chain_complex_of (F : W₁ ⥤ W₂) [F.PreservesZeroMorphisms] (X : α → W₁)
    (d : ∀ n, X (n + 1) ⟶ X n) (sq : ∀ n, d (n + 1) ≫ d n = 0) :
    (F.mapHomologicalComplex _).obj (ChainComplex.of X d sq) =
      ChainComplex.of (fun n => F.obj (X n)) (fun n => F.map (d n)) fun n => by
        /-
          ι : Type u_1
          V : Type u
          inst✝¹¹ : CategoryTheory.Category.{v, u} V
          inst✝¹⁰ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁹ : CategoryTheory.Category.{?u.68928, u_2} W
          inst✝⁸ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.68954, u_3} W₁
          inst✝⁶ : CategoryTheory.Category.{?u.68958, u_4} W₂
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c : ComplexShape ι
          C D : HomologicalComplex V c
          f : Quiver.Hom C D
          i : ι
          α : Type u_5
          inst✝³ : AddRightCancelSemigroup α
          inst✝² : One α
          inst✝¹ : DecidableEq α
          F : CategoryTheory.Functor W₁ W₂
          inst✝ : F.PreservesZeroMorphisms
          X : α → W₁
          d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
          sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
          n : α
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => F.map (d n)) (HAdd.hAdd n  …
        -/
        rw [← F.map_comp, sq n, Functor.map_zero] := by
        /-
          🎉 no goals
        -/
  /-
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_7, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    α : Type u_5
    inst✝³ : AddRightCancelSemigroup α
    inst✝² : One α
    inst✝¹ : DecidableEq α
    F : CategoryTheory.Functor W₁ W₂
    inst✝ : F.PreservesZeroMorphisms
    X : α → W₁
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    ⊢ Eq ((F.mapHomologicalComplex (ComplexShape.down α)).obj (ChainComplex.of X d …
  -/
  refine HomologicalComplex.ext rfl ?_
  /-
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_7, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    α : Type u_5
    inst✝³ : AddRightCancelSemigroup α
    inst✝² : One α
    inst✝¹ : DecidableEq α
    F : CategoryTheory.Functor W₁ W₂
    inst✝ : F.PreservesZeroMorphisms
    X : α → W₁
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    ⊢ ∀ (i j : α), (ComplexShape.down α).Rel i j → Eq (CategoryTheory.CategoryStru …
  -/
  rintro i j (rfl : j + 1 = i)
  simp only [CategoryTheory.Functor.mapHomologicalComplex_obj_d, of_d, eqToHom_refl, comp_id,
    id_comp]


instance (W : Type*) [Category W] [Preadditive W] [HasZeroObject W] [DecidableEq ι] (j : ι) :
    (single W c j).Additive where
                          /-
                            ι : Type u_1
                            V : Type u
                            inst✝¹³ : CategoryTheory.Category.{v, u} V
                            inst✝¹² : CategoryTheory.Preadditive V
                            W✝ : Type u_2
                            inst✝¹¹ : CategoryTheory.Category.{?u.73625, u_2} W✝
                            inst✝¹⁰ : CategoryTheory.Preadditive W✝
                            W₁ : Type u_3
                            W₂ : Type u_4
                            inst✝⁹ : CategoryTheory.Category.{?u.73651, u_3} W₁
                            inst✝⁸ : CategoryTheory.Category.{?u.73655, u_4} W₂
                            inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms W₁
                            inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms W₂
                            c : ComplexShape ι
                            C D : HomologicalComplex V c
                            f✝ : Quiver.Hom C D
                            i : ι
                            inst✝⁵ : CategoryTheory.Limits.HasZeroObject W₁
                            inst✝⁴ : CategoryTheory.Limits.HasZeroObject W₂
                            W : Type u_5
                            inst✝³ : CategoryTheory.Category.{u_6, u_5} W
                            inst✝² : CategoryTheory.Preadditive W
                            inst✝¹ : CategoryTheory.Limits.HasZeroObject W
                            inst✝ : DecidableEq ι
                            j : ι
                            x✝¹ x✝ : W
                            f g : Quiver.Hom x✝¹ x✝
                            ⊢ Eq ((HomologicalComplex.single W c j).map (HAdd.hAdd f g)) (HAdd.hAdd ((Homo …
                          -/
  map_add {_ _ f g} := by ext; simp [single]
                               /-
                                 🎉 no goals
                               -/


/-- Turning an object into a complex supported at `j` then applying a functor is
the same as applying the functor then forming the complex.
-/
noncomputable def singleMapHomologicalComplex (j : ι) :
    single W₁ c j ⋙ F.mapHomologicalComplex _ ≅ F ⋙ single W₂ c j :=
  NatIso.ofComponents
    (fun X =>
                                                             /-
                                                               ι : Type u_1
                                                               V : Type u
                                                               inst✝¹¹ : CategoryTheory.Category.{v, u} V
                                                               inst✝¹⁰ : CategoryTheory.Preadditive V
                                                               W : Type u_2
                                                               inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
                                                               inst✝⁸ : CategoryTheory.Preadditive W
                                                               W₁ : Type u_3
                                                               W₂ : Type u_4
                                                               inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
                                                               inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
                                                               inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
                                                               inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                                               c✝ : ComplexShape ι
                                                               C D : HomologicalComplex V c✝
                                                               f : Quiver.Hom C D
                                                               i✝ : ι
                                                               inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
                                                               inst✝² : CategoryTheory.Limits.HasZeroObject W₂
                                                               F : CategoryTheory.Functor W₁ W₂
                                                               inst✝¹ : F.PreservesZeroMorphisms
                                                               c : ComplexShape ι
                                                               inst✝ : DecidableEq ι
                                                               j : ι
                                                               X : W₁
                                                               i : ι
                                                               h : Eq i j
                                                               ⊢ Eq ((((HomologicalComplex.single W₁ c j).comp (F.mapHomologicalComplex c)).o …
                                                             -/
      { hom := { f := fun i => if h : i = j then eqToHom (by simp [h]) else 0 }
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                             /-
                                                               ι : Type u_1
                                                               V : Type u
                                                               inst✝¹¹ : CategoryTheory.Category.{v, u} V
                                                               inst✝¹⁰ : CategoryTheory.Preadditive V
                                                               W : Type u_2
                                                               inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
                                                               inst✝⁸ : CategoryTheory.Preadditive W
                                                               W₁ : Type u_3
                                                               W₂ : Type u_4
                                                               inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
                                                               inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
                                                               inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
                                                               inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
                                                               c✝ : ComplexShape ι
                                                               C D : HomologicalComplex V c✝
                                                               f : Quiver.Hom C D
                                                               i✝ : ι
                                                               inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
                                                               inst✝² : CategoryTheory.Limits.HasZeroObject W₂
                                                               F : CategoryTheory.Functor W₁ W₂
                                                               inst✝¹ : F.PreservesZeroMorphisms
                                                               c : ComplexShape ι
                                                               inst✝ : DecidableEq ι
                                                               j : ι
                                                               X : W₁
                                                               i : ι
                                                               h : Eq i j
                                                               ⊢ Eq (((F.comp (HomologicalComplex.single W₂ c j)).obj X).X i) ((((Homological …
                                                             -/
        inv := { f := fun i => if h : i = j then eqToHom (by simp [h]) else 0 }
                                                             /-
                                                               🎉 no goals
                                                             -/
        hom_inv_id := by
          /-
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun i => dite (Eq i j) (fun h  …
          -/
          ext i
          /-
            case h
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i✝ : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            i : ι
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun i => dite (Eq i j) (fun h …
          -/
          dsimp
          /-
            case h
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i✝ : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun h => CategoryTheo …
          -/
          split_ifs with h
            /-
              case pos
              ι : Type u_1
              V : Type u
              inst✝¹¹ : CategoryTheory.Category.{v, u} V
              inst✝¹⁰ : CategoryTheory.Preadditive V
              W : Type u_2
              inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
              inst✝⁸ : CategoryTheory.Preadditive W
              W₁ : Type u_3
              W₂ : Type u_4
              inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
              inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
              inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
              inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
              c✝ : ComplexShape ι
              C D : HomologicalComplex V c✝
              f : Quiver.Hom C D
              i✝ : ι
              inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
              inst✝² : CategoryTheory.Limits.HasZeroObject W₂
              F : CategoryTheory.Functor W₁ W₂
              inst✝¹ : F.PreservesZeroMorphisms
              c : ComplexShape ι
              inst✝ : DecidableEq ι
              j : ι
              X : W₁
              i : ι
              h : Eq i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
            -/
          · simp [h]
            /-
              🎉 no goals
            -/
          · rw [zero_comp, ← F.map_id,
              (isZero_single_obj_X c j X _ h).eq_of_src (𝟙 _) 0, F.map_zero]
        inv_hom_id := by
          /-
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun i => dite (Eq i j) (fun h  …
          -/
          ext i
          /-
            case h
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i✝ : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            i : ι
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun i => dite (Eq i j) (fun h …
          -/
          dsimp
          /-
            case h
            ι : Type u_1
            V : Type u
            inst✝¹¹ : CategoryTheory.Category.{v, u} V
            inst✝¹⁰ : CategoryTheory.Preadditive V
            W : Type u_2
            inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
            inst✝⁸ : CategoryTheory.Preadditive W
            W₁ : Type u_3
            W₂ : Type u_4
            inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
            inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
            c✝ : ComplexShape ι
            C D : HomologicalComplex V c✝
            f : Quiver.Hom C D
            i✝ : ι
            inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
            inst✝² : CategoryTheory.Limits.HasZeroObject W₂
            F : CategoryTheory.Functor W₁ W₂
            inst✝¹ : F.PreservesZeroMorphisms
            c : ComplexShape ι
            inst✝ : DecidableEq ι
            j : ι
            X : W₁
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun h => CategoryTheo …
          -/
          split_ifs with h
            /-
              case pos
              ι : Type u_1
              V : Type u
              inst✝¹¹ : CategoryTheory.Category.{v, u} V
              inst✝¹⁰ : CategoryTheory.Preadditive V
              W : Type u_2
              inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
              inst✝⁸ : CategoryTheory.Preadditive W
              W₁ : Type u_3
              W₂ : Type u_4
              inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
              inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
              inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
              inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
              c✝ : ComplexShape ι
              C D : HomologicalComplex V c✝
              f : Quiver.Hom C D
              i✝ : ι
              inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
              inst✝² : CategoryTheory.Limits.HasZeroObject W₂
              F : CategoryTheory.Functor W₁ W₂
              inst✝¹ : F.PreservesZeroMorphisms
              c : ComplexShape ι
              inst✝ : DecidableEq ι
              j : ι
              X : W₁
              i : ι
              h : Eq i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
            -/
          · simp [h]
            /-
              🎉 no goals
            -/
            /-
              case neg
              ι : Type u_1
              V : Type u
              inst✝¹¹ : CategoryTheory.Category.{v, u} V
              inst✝¹⁰ : CategoryTheory.Preadditive V
              W : Type u_2
              inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
              inst✝⁸ : CategoryTheory.Preadditive W
              W₁ : Type u_3
              W₂ : Type u_4
              inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
              inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
              inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
              inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
              c✝ : ComplexShape ι
              C D : HomologicalComplex V c✝
              f : Quiver.Hom C D
              i✝ : ι
              inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
              inst✝² : CategoryTheory.Limits.HasZeroObject W₂
              F : CategoryTheory.Functor W₁ W₂
              inst✝¹ : F.PreservesZeroMorphisms
              c : ComplexShape ι
              inst✝ : DecidableEq ι
              j : ι
              X : W₁
              i : ι
              h : Not (Eq i j)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
            -/
          · apply (isZero_single_obj_X c j _ _ h).eq_of_src })
            /-
              🎉 no goals
            -/
    fun f => by
      /-
        ι : Type u_1
        V : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} V
        inst✝¹⁰ : CategoryTheory.Preadditive V
        W : Type u_2
        inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
        inst✝⁸ : CategoryTheory.Preadditive W
        W₁ : Type u_3
        W₂ : Type u_4
        inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
        inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
        c✝ : ComplexShape ι
        C D : HomologicalComplex V c✝
        f✝ : Quiver.Hom C D
        i : ι
        inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
        inst✝² : CategoryTheory.Limits.HasZeroObject W₂
        F : CategoryTheory.Functor W₁ W₂
        inst✝¹ : F.PreservesZeroMorphisms
        c : ComplexShape ι
        inst✝ : DecidableEq ι
        j : ι
        X✝ Y✝ : W₁
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.single W₁ c j). …
      -/
      ext i
      /-
        case h
        ι : Type u_1
        V : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} V
        inst✝¹⁰ : CategoryTheory.Preadditive V
        W : Type u_2
        inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
        inst✝⁸ : CategoryTheory.Preadditive W
        W₁ : Type u_3
        W₂ : Type u_4
        inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
        inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
        c✝ : ComplexShape ι
        C D : HomologicalComplex V c✝
        f✝ : Quiver.Hom C D
        i✝ : ι
        inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
        inst✝² : CategoryTheory.Limits.HasZeroObject W₂
        F : CategoryTheory.Functor W₁ W₂
        inst✝¹ : F.PreservesZeroMorphisms
        c : ComplexShape ι
        inst✝ : DecidableEq ι
        j : ι
        X✝ Y✝ : W₁
        f : Quiver.Hom X✝ Y✝
        i : ι
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((HomologicalComplex.single W₁ c j) …
      -/
      dsimp
      /-
        case h
        ι : Type u_1
        V : Type u
        inst✝¹¹ : CategoryTheory.Category.{v, u} V
        inst✝¹⁰ : CategoryTheory.Preadditive V
        W : Type u_2
        inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
        inst✝⁸ : CategoryTheory.Preadditive W
        W₁ : Type u_3
        W₂ : Type u_4
        inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
        inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
        inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
        c✝ : ComplexShape ι
        C D : HomologicalComplex V c✝
        f✝ : Quiver.Hom C D
        i✝ : ι
        inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
        inst✝² : CategoryTheory.Limits.HasZeroObject W₂
        F : CategoryTheory.Functor W₁ W₂
        inst✝¹ : F.PreservesZeroMorphisms
        c : ComplexShape ι
        inst✝ : DecidableEq ι
        j : ι
        X✝ Y✝ : W₁
        f : Quiver.Hom X✝ Y✝
        i : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (((HomologicalComplex.single W …
      -/
      split_ifs with h
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝¹¹ : CategoryTheory.Category.{v, u} V
          inst✝¹⁰ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
          inst✝⁸ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
          inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C D : HomologicalComplex V c✝
          f✝ : Quiver.Hom C D
          i✝ : ι
          inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
          inst✝² : CategoryTheory.Limits.HasZeroObject W₂
          F : CategoryTheory.Functor W₁ W₂
          inst✝¹ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          inst✝ : DecidableEq ι
          j : ι
          X✝ Y✝ : W₁
          f : Quiver.Hom X✝ Y✝
          i : ι
          h : Eq i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (((HomologicalComplex.single W …
        -/
      · subst h
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝¹¹ : CategoryTheory.Category.{v, u} V
          inst✝¹⁰ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
          inst✝⁸ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
          inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C D : HomologicalComplex V c✝
          f✝ : Quiver.Hom C D
          i✝ : ι
          inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
          inst✝² : CategoryTheory.Limits.HasZeroObject W₂
          F : CategoryTheory.Functor W₁ W₂
          inst✝¹ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          inst✝ : DecidableEq ι
          X✝ Y✝ : W₁
          f : Quiver.Hom X✝ Y✝
          i : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (((HomologicalComplex.single W …
        -/
        simp [single_map_f_self, singleObjXSelf, singleObjXIsoOfEq, eqToHom_map]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          V : Type u
          inst✝¹¹ : CategoryTheory.Category.{v, u} V
          inst✝¹⁰ : CategoryTheory.Preadditive V
          W : Type u_2
          inst✝⁹ : CategoryTheory.Category.{?u.97874, u_2} W
          inst✝⁸ : CategoryTheory.Preadditive W
          W₁ : Type u_3
          W₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.97900, u_3} W₁
          inst✝⁶ : CategoryTheory.Category.{?u.97904, u_4} W₂
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
          c✝ : ComplexShape ι
          C D : HomologicalComplex V c✝
          f✝ : Quiver.Hom C D
          i✝ : ι
          inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
          inst✝² : CategoryTheory.Limits.HasZeroObject W₂
          F : CategoryTheory.Functor W₁ W₂
          inst✝¹ : F.PreservesZeroMorphisms
          c : ComplexShape ι
          inst✝ : DecidableEq ι
          j : ι
          X✝ Y✝ : W₁
          f : Quiver.Hom X✝ Y✝
          i : ι
          h : Not (Eq i j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (((HomologicalComplex.single W …
        -/
      · apply (isZero_single_obj_X c j _ _ h).eq_of_tgt
        /-
          🎉 no goals
        -/


@[simp]
theorem singleMapHomologicalComplex_hom_app_self (j : ι) (X : W₁) :
    ((singleMapHomologicalComplex F c j).hom.app X).f j =
      F.map (singleObjXSelf c j X).hom ≫ (singleObjXSelf c j (F.obj X)).inv := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
    inst✝² : CategoryTheory.Limits.HasZeroObject W₂
    F : CategoryTheory.Functor W₁ W₂
    inst✝¹ : F.PreservesZeroMorphisms
    c : ComplexShape ι
    inst✝ : DecidableEq ι
    j : ι
    X : W₁
    ⊢ Eq (((HomologicalComplex.singleMapHomologicalComplex F c j).hom.app X).f j)  …
  -/
  simp [singleMapHomologicalComplex, singleObjXSelf, singleObjXIsoOfEq, eqToHom_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleMapHomologicalComplex_hom_app_ne {i j : ι} (h : i ≠ j) (X : W₁) :
    ((singleMapHomologicalComplex F c j).hom.app X).f i = 0 := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
    inst✝² : CategoryTheory.Limits.HasZeroObject W₂
    F : CategoryTheory.Functor W₁ W₂
    inst✝¹ : F.PreservesZeroMorphisms
    c : ComplexShape ι
    inst✝ : DecidableEq ι
    i j : ι
    h : Ne i j
    X : W₁
    ⊢ Eq (((HomologicalComplex.singleMapHomologicalComplex F c j).hom.app X).f i) 0
  -/
  simp [singleMapHomologicalComplex, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleMapHomologicalComplex_inv_app_self (j : ι) (X : W₁) :
    ((singleMapHomologicalComplex F c j).inv.app X).f j =
      (singleObjXSelf c j (F.obj X)).hom ≫ F.map (singleObjXSelf c j X).inv := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
    inst✝² : CategoryTheory.Limits.HasZeroObject W₂
    F : CategoryTheory.Functor W₁ W₂
    inst✝¹ : F.PreservesZeroMorphisms
    c : ComplexShape ι
    inst✝ : DecidableEq ι
    j : ι
    X : W₁
    ⊢ Eq (((HomologicalComplex.singleMapHomologicalComplex F c j).inv.app X).f j)  …
  -/
  simp [singleMapHomologicalComplex, singleObjXSelf, singleObjXIsoOfEq, eqToHom_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleMapHomologicalComplex_inv_app_ne {i j : ι} (h : i ≠ j) (X : W₁) :
    ((singleMapHomologicalComplex F c j).inv.app X).f i = 0 := by
  /-
    ι : Type u_1
    W₁ : Type u_3
    W₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} W₁
    inst✝⁶ : CategoryTheory.Category.{u_5, u_4} W₂
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms W₁
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms W₂
    inst✝³ : CategoryTheory.Limits.HasZeroObject W₁
    inst✝² : CategoryTheory.Limits.HasZeroObject W₂
    F : CategoryTheory.Functor W₁ W₂
    inst✝¹ : F.PreservesZeroMorphisms
    c : ComplexShape ι
    inst✝ : DecidableEq ι
    i j : ι
    h : Ne i j
    X : W₁
    ⊢ Eq (((HomologicalComplex.singleMapHomologicalComplex F c j).inv.app X).f i) 0
  -/
  simp [singleMapHomologicalComplex, h]
  /-
    🎉 no goals
  -/


