@[simp, reassoc]
theorem p_comp_d : P.p.f n ≫ f.f.f n = f.f.f n :=
  HomologicalComplex.congr_hom (p_comp f) n


@[simp, reassoc]
theorem comp_p_d : f.f.f n ≫ Q.p.f n = f.f.f n :=
  HomologicalComplex.congr_hom (comp_p f) n


@[reassoc]
theorem p_comm_f : P.p.f n ≫ f.f.f n = f.f.f n ≫ Q.p.f n :=
  HomologicalComplex.congr_hom (p_comm f) n


@[simp, reassoc]
theorem p_idem : P.p.f n ≫ P.p.f n = P.p.f n :=
  HomologicalComplex.congr_hom P.idem n


/-- The functor `Karoubi (HomologicalComplex C c) ⥤ HomologicalComplex (Karoubi C) c`,
on objects. -/
@[simps]
def obj (P : Karoubi (HomologicalComplex C c)) : HomologicalComplex (Karoubi C) c where
  X n :=
    ⟨P.X.X n, P.p.f n, by
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.3974, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
        n : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.f n) (P.p.f n)) (P.p.f n)
      -/
      simpa only [HomologicalComplex.comp_f] using HomologicalComplex.congr_hom P.idem n⟩
      /-
        🎉 no goals
      -/
  d i j := { f := P.p.f i ≫ P.X.d i j }
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{?u.3974, u_1} C
                        inst✝ : CategoryTheory.Preadditive C
                        ι : Type u_2
                        c : ComplexShape ι
                        P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
                        i j : ι
                        hij : Not (c.Rel i j)
                        ⊢ Eq ((fun i j => { f := CategoryTheory.CategoryStruct.comp (P.p.f i) (P.X.d i …
                      -/
  shape i j hij := by simp only [hom_eq_zero_iff, P.X.shape i j hij, Limits.comp_zero]; aesop_cat
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


/-- The functor `Karoubi (HomologicalComplex C c) ⥤ HomologicalComplex (Karoubi C) c`,
on morphisms. -/
@[simps]
def map {P Q : Karoubi (HomologicalComplex C c)} (f : P ⟶ Q) : obj P ⟶ obj Q where
  f n :=
    { f := f.f.f n }


/-- The functor `Karoubi (HomologicalComplex C c) ⥤ HomologicalComplex (Karoubi C) c`. -/
@[simps]
def functor : Karoubi (HomologicalComplex C c) ⥤ HomologicalComplex (Karoubi C) c where
  obj := Functor.obj
  map f := Functor.map f


/-- The functor `HomologicalComplex (Karoubi C) c ⥤ Karoubi (HomologicalComplex C c)`,
on objects -/
@[simps]
def obj (K : HomologicalComplex (Karoubi C) c) : Karoubi (HomologicalComplex C c) where
  X :=
    { X := fun n => (K.X n).X
      d := fun i j => (K.d i j).f
      shape := fun i j hij => hom_eq_zero_iff.mp (K.shape i j hij)
      d_comp_d' := fun i j k _ _ => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.10234, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          K : HomologicalComplex (CategoryTheory.Idempotents.Karoubi C) c
          i j k : ι
          x✝¹ : c.Rel i j
          x✝ : c.Rel j k
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (K.d i j).f) i j) ((fun  …
        -/
        simpa only [comp_f] using hom_eq_zero_iff.mp (K.d_comp_d i j k) }
        /-
          🎉 no goals
        -/
  p := { f := fun n => (K.X n).p }


/-- The functor `HomologicalComplex (Karoubi C) c ⥤ Karoubi (HomologicalComplex C c)`,
on morphisms -/
@[simps]
def map {K L : HomologicalComplex (Karoubi C) c} (f : K ⟶ L) : obj K ⟶ obj L where
  f :=
    { f := fun n => (f.f n).f
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.14345, u_1} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   ι : Type u_2
                                   c : ComplexShape ι
                                   K L : HomologicalComplex (CategoryTheory.Idempotents.Karoubi C) c
                                   f : Quiver.Hom K L
                                   i j : ι
                                   hij : c.Rel i j
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => (f.f n).f) i) ((CategoryTh …
                                 -/
      comm' := fun i j hij => by simpa only [comp_f] using hom_ext_iff.mp (f.comm' i j hij) }
                                 /-
                                   🎉 no goals
                                 -/


/-- The functor `HomologicalComplex (Karoubi C) c ⥤ Karoubi (HomologicalComplex C c)`. -/
@[simps]
def inverse : HomologicalComplex (Karoubi C) c ⥤ Karoubi (HomologicalComplex C c) where
  obj := Inverse.obj
  map f := Inverse.map f


/-- The counit isomorphism of the equivalence
`Karoubi (HomologicalComplex C c) ≌ HomologicalComplex (Karoubi C) c`. -/
@[simps!]
def counitIso : inverse ⋙ functor ≅ 𝟭 (HomologicalComplex (Karoubi C) c) :=
                                                            /-
                                                              C : Type u_1
                                                              inst✝¹ : CategoryTheory.Category.{?u.18417, u_1} C
                                                              inst✝ : CategoryTheory.Preadditive C
                                                              ι : Type u_2
                                                              c : ComplexShape ι
                                                              P : HomologicalComplex (CategoryTheory.Idempotents.Karoubi C) c
                                                              ⊢ Eq ((CategoryTheory.Idempotents.KaroubiHomologicalComplexEquivalence.inverse …
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  eqToIso (Functor.ext (fun P => HomologicalComplex.ext (by aesop_cat) (by aesop_cat))
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.18417, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          ⊢ ∀ (X Y : HomologicalComplex (CategoryTheory.Idempotents.Karoubi C) c) (f : Q …
        -/
    (by aesop_cat))
        /-
          🎉 no goals
        -/


/-- The unit isomorphism of the equivalence
`Karoubi (HomologicalComplex C c) ≌ HomologicalComplex (Karoubi C) c`. -/
@[simps]
def unitIso : 𝟭 (Karoubi (HomologicalComplex C c)) ≅ functor ⋙ inverse where
  hom :=
    { app := fun P =>
        { f :=
            { f := fun n => P.p.f n
              comm' := fun i j _ => by
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
                  inst✝ : CategoryTheory.Preadditive C
                  ι : Type u_2
                  c : ComplexShape ι
                  P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
                  i j : ι
                  x✝ : c.Rel i j
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => P.p.f n) i) (((CategoryThe …
                -/
                dsimp
                simp only [HomologicalComplex.Hom.comm, HomologicalComplex.Hom.comm_assoc,
                  HomologicalComplex.p_idem] }
          comm := by
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              ⊢ Eq { f := fun n => P.p.f n, comm' := ⋯ } (CategoryTheory.CategoryStruct.comp …
            -/
            ext n
            /-
              case h
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              n : ι
              ⊢ Eq ({ f := fun n => P.p.f n, comm' := ⋯ }.f n) ((CategoryTheory.CategoryStru …
            -/
            dsimp
            /-
              case h
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              n : ι
              ⊢ Eq (P.p.f n) (CategoryTheory.CategoryStruct.comp (P.p.f n) (CategoryTheory.C …
            -/
            simp only [HomologicalComplex.p_idem] }
            /-
              🎉 no goals
            -/
      naturality := fun P Q φ => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          P Q : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
          φ : Quiver.Hom P Q
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
        -/
        ext
        /-
          case h.h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          P Q : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
          φ : Quiver.Hom P Q
          i✝ : ι
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categor …
        -/
        dsimp
        simp only [comp_f, HomologicalComplex.comp_f, HomologicalComplex.comp_p_d, Inverse.map_f_f,
          Functor.map_f_f, HomologicalComplex.p_comp_d] }
  inv :=
    { app := fun P =>
        { f :=
            { f := fun n => P.p.f n
              comm' := fun i j _ => by
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
                  inst✝ : CategoryTheory.Preadditive C
                  ι : Type u_2
                  c : ComplexShape ι
                  P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
                  i j : ι
                  x✝ : c.Rel i j
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => P.p.f n) i) (((CategoryThe …
                -/
                dsimp
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
                  inst✝ : CategoryTheory.Preadditive C
                  ι : Type u_2
                  c : ComplexShape ι
                  P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
                  i j : ι
                  x✝ : c.Rel i j
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.p.f i) (P.X.d i j)) (CategoryTheor …
                -/
                simp only [HomologicalComplex.Hom.comm, assoc, HomologicalComplex.p_idem] }
                /-
                  🎉 no goals
                -/
          comm := by
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              ⊢ Eq { f := fun n => P.p.f n, comm' := ⋯ } (CategoryTheory.CategoryStruct.comp …
            -/
            ext n
            /-
              case h
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              n : ι
              ⊢ Eq ({ f := fun n => P.p.f n, comm' := ⋯ }.f n) ((CategoryTheory.CategoryStru …
            -/
            dsimp
            /-
              case h
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
              inst✝ : CategoryTheory.Preadditive C
              ι : Type u_2
              c : ComplexShape ι
              P : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
              n : ι
              ⊢ Eq (P.p.f n) (CategoryTheory.CategoryStruct.comp (P.p.f n) (CategoryTheory.C …
            -/
            simp only [HomologicalComplex.p_idem] }
            /-
              🎉 no goals
            -/
      naturality := fun P Q φ => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          P Q : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
          φ : Quiver.Hom P Q
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.KaroubiH …
        -/
        ext
        /-
          case h.h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          P Q : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
          φ : Quiver.Hom P Q
          i✝ : ι
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.Karoubi …
        -/
        dsimp
        simp only [comp_f, HomologicalComplex.comp_f, Inverse.map_f_f, Functor.map_f_f,
          HomologicalComplex.comp_p_d, HomologicalComplex.p_comp_d] }
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun P => { f := { f := fun n …
    -/
    ext
    /-
      case w.h.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      x✝ : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
      i✝ : ι
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp { app := fun P => { f := { f := fun …
    -/
    dsimp
    /-
      case w.h.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      x✝ : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
      i✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (x✝.p.f i✝) (x✝.p.f i✝)) (x✝.p.f i✝)
    -/
    simp only [HomologicalComplex.p_idem, comp_f, HomologicalComplex.comp_f, _root_.id_eq]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun P => { f := { f := fun n …
    -/
    ext
    /-
      case w.h.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.28447, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      x✝ : CategoryTheory.Idempotents.Karoubi (HomologicalComplex C c)
      i✝ : ι
      ⊢ Eq (((CategoryTheory.CategoryStruct.comp { app := fun P => { f := { f := fun …
    -/
    dsimp
    simp only [HomologicalComplex.p_idem, comp_f, HomologicalComplex.comp_f, _root_.id_eq,
      Inverse.obj_p_f, Functor.obj_X_p]


/-- The equivalence `Karoubi (HomologicalComplex C c) ≌ HomologicalComplex (Karoubi C) c`. -/
@[simps]
def karoubiHomologicalComplexEquivalence :
    Karoubi (HomologicalComplex C c) ≌ HomologicalComplex (Karoubi C) c where
  functor := KaroubiHomologicalComplexEquivalence.functor
  inverse := KaroubiHomologicalComplexEquivalence.inverse
  unitIso := KaroubiHomologicalComplexEquivalence.unitIso
  counitIso := KaroubiHomologicalComplexEquivalence.counitIso


/-- The equivalence `Karoubi (ChainComplex C α) ≌ ChainComplex (Karoubi C) α`. -/
@[simps!]
def karoubiChainComplexEquivalence : Karoubi (ChainComplex C α) ≌ ChainComplex (Karoubi C) α :=
  karoubiHomologicalComplexEquivalence C (ComplexShape.down α)


/-- The equivalence `Karoubi (CochainComplex C α) ≌ CochainComplex (Karoubi C) α`. -/
@[simps!]
def karoubiCochainComplexEquivalence :
    Karoubi (CochainComplex C α) ≌ CochainComplex (Karoubi C) α :=
  karoubiHomologicalComplexEquivalence C (ComplexShape.up α)


instance [IsIdempotentComplete C] : IsIdempotentComplete (HomologicalComplex C c) := by
  rw [isIdempotentComplete_iff_of_equivalence
      ((toKaroubiEquivalence C).mapHomologicalComplex c),
    ← isIdempotentComplete_iff_of_equivalence (karoubiHomologicalComplexEquivalence C c)]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    α : Type u_3
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : CategoryTheory.IsIdempotentComplete C
    ⊢ CategoryTheory.IsIdempotentComplete (CategoryTheory.Idempotents.Karoubi (Hom …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


