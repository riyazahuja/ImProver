/-- A cone in `HomologicalComplex C c` is limit if the induced cones obtained
by applying `eval C c i : HomologicalComplex C c ⥤ C` for all `i` are limit. -/
def isLimitOfEval (s : Cone F)
    (hs : ∀ (i : ι), IsLimit ((eval C c i).mapCone s)) : IsLimit s where
  lift t :=
    { f := fun i => (hs i).lift ((eval C c i).mapCone t)
      comm' := fun i i' _ => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (hs i).lift ((HomologicalC …
        -/
        apply IsLimit.hom_ext (hs i')
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
        -/
        intro j
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        have eq := fun k => (hs k).fac ((eval C c k).mapCone t)
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp ((hs k).lift (( …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Functor.mapCone_π_app, eval_map] at eq
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp ((hs k).lift (( …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Functor.mapCone_π_app, eval_map, assoc]
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.161, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cone F
          hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
          t : CategoryTheory.Limits.Cone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp ((hs k).lift (( …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((hs i).lift ((HomologicalComplex.eva …
        -/
        rw [eq i', ← Hom.comm, reassoc_of% (eq i), Hom.comm] }
        /-
          🎉 no goals
        -/
  fac t j := by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun t => { f := fun i => (hs i).lif …
    -/
    ext i
    /-
      case h
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      j : J
      i : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun t => { f := fun i => (hs i).li …
    -/
    apply (hs i).fac
    /-
      🎉 no goals
    -/
  uniq t m hm := by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      m : Quiver.Hom t.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (s.π.app j)) (t.π.app …
      ⊢ Eq m ((fun t => { f := fun i => (hs i).lift ((HomologicalComplex.eval C c i) …
    -/
    ext i
    /-
      case h
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      m : Quiver.Hom t.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (s.π.app j)) (t.π.app …
      i : ι
      ⊢ Eq (m.f i) (((fun t => { f := fun i => (hs i).lift ((HomologicalComplex.eval …
    -/
    apply (hs i).uniq ((eval C c i).mapCone t)
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      m : Quiver.Hom t.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (s.π.app j)) (t.π.app …
      i : ι
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (m.f i) (((HomologicalComp …
    -/
    intro j
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      m : Quiver.Hom t.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (s.π.app j)) (t.π.app …
      i : ι
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (m.f i) (((HomologicalComplex.eval C  …
    -/
    dsimp
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.161, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.165, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cone F
      hs : (i : ι) → CategoryTheory.Limits.IsLimit ((HomologicalComplex.eval C c i). …
      t : CategoryTheory.Limits.Cone F
      m : Quiver.Hom t.pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (s.π.app j)) (t.π.app …
      i : ι
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (m.f i) ((s.π.app j).f i)) ((t.π.app  …
    -/
    simp only [← comp_f, hm]
    /-
      🎉 no goals
    -/


/-- A cone for a functor `F : J ⥤ HomologicalComplex C c` which is given in degree `n` by
the limit `F ⋙ eval C c n`. -/
@[simps]
noncomputable def coneOfHasLimitEval : Cone F where
  pt :=
    { X := fun n => limit (F ⋙ eval C c n)
      d := fun n m => limMap { app := fun j => (F.obj j).d n m }
      shape := fun {n m} h => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          n m : ι
          h : Not (c.Rel n m)
          ⊢ Eq ((fun n m => CategoryTheory.Limits.limMap { app := fun j => (F.obj j).d n …
        -/
        ext j
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n m => CategoryTheory.Limits.li …
        -/
        rw [limMap_π]
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
        -/
        dsimp
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (F.com …
        -/
        rw [(F.obj j).shape _ _ h, comp_zero, zero_comp] }
        /-
          🎉 no goals
        -/
  π :=
    { app := fun j => { f := fun _ => limit.π _ j }
      naturality := fun i j φ => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          i j : J
          φ : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext n
        /-
          case h
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          i j : J
          φ : Quiver.Hom i j
          n : ι
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).o …
        -/
        dsimp
        /-
          case h
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.6046, u_1} C
          inst✝² : CategoryTheory.Category.{?u.6050, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasLimit (F.comp (HomologicalComplex. …
          i j : J
          φ : Quiver.Hom i j
          n : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
        -/
        erw [limit.w, id_comp] }
        /-
          🎉 no goals
        -/


/-- The cone `coneOfHasLimitEval F` is limit. -/
noncomputable def isLimitConeOfHasLimitEval : IsLimit (coneOfHasLimitEval F) :=
  isLimitOfEval _ _ (fun _ => limit.isLimit _)


instance : HasLimit F := ⟨⟨⟨_, isLimitConeOfHasLimitEval F⟩⟩⟩


noncomputable instance (n : ι) : PreservesLimit F (eval C c n) :=
  preservesLimit_of_preserves_limit_cone (isLimitConeOfHasLimitEval F) (limit.isLimit _)


instance [HasLimitsOfShape J C] : HasLimitsOfShape J (HomologicalComplex C c) := ⟨inferInstance⟩


noncomputable instance [HasLimitsOfShape J C] (n : ι) :
  PreservesLimitsOfShape J (eval C c n) := ⟨inferInstance⟩


instance [HasFiniteLimits C] : HasFiniteLimits (HomologicalComplex C c) :=
  ⟨fun _ _ => inferInstance⟩


noncomputable instance [HasFiniteLimits C] (n : ι) :
  PreservesFiniteLimits (eval C c n) := ⟨fun _ _ _ => inferInstance⟩


instance [HasFiniteLimits C] {K L : HomologicalComplex C c} (φ : K ⟶ L) [Mono φ] (n : ι) :
    Mono (φ.f n) := by
  /-
    C : Type u_1
    ι : Type u_2
    J : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.36064, u_3} J
    c : ComplexShape ι
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    inst✝ : CategoryTheory.Mono φ
    n : ι
    ⊢ CategoryTheory.Mono (φ.f n)
  -/
  change Mono ((HomologicalComplex.eval C c n).map φ)
  /-
    C : Type u_1
    ι : Type u_2
    J : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.36064, u_3} J
    c : ComplexShape ι
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    inst✝ : CategoryTheory.Mono φ
    n : ι
    ⊢ CategoryTheory.Mono ((HomologicalComplex.eval C c n).map φ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A cocone in `HomologicalComplex C c` is colimit if the induced cocones obtained
by applying `eval C c i : HomologicalComplex C c ⥤ C` for all `i` are colimit. -/
def isColimitOfEval (s : Cocone F)
    (hs : ∀ (i : ι), IsColimit ((eval C c i).mapCocone s)) : IsColimit s where
  desc t :=
    { f := fun i => (hs i).desc ((eval C c i).mapCocone t)
      comm' := fun i i' _ => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (hs i).desc ((HomologicalC …
        -/
        apply IsColimit.hom_ext (hs i)
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval …
        -/
        intro j
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval C c i).map …
        -/
        have eq := fun k => (hs k).fac ((eval C c k).mapCocone t)
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp (((HomologicalC …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval C c i).map …
        -/
        simp only [Functor.mapCocone_ι_app, eval_map] at eq
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp ((s.ι.app j).f  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval C c i).map …
        -/
        simp only [Functor.mapCocone_ι_app, eval_map, assoc]
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
          c : ComplexShape ι
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          s : CategoryTheory.Limits.Cocone F
          hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
          t : CategoryTheory.Limits.Cocone F
          i i' : ι
          x✝ : c.Rel i i'
          j : J
          eq : ∀ (k : ι) (j : J), Eq (CategoryTheory.CategoryStruct.comp ((s.ι.app j).f  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.ι.app j).f i) (CategoryTheory.Cat …
        -/
        rw [reassoc_of% (eq i), Hom.comm_assoc, eq i', Hom.comm] }
        /-
          🎉 no goals
        -/
  fac t j := by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) ((fun t => { f := fun i = …
    -/
    ext i
    /-
      case h
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      j : J
      i : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (s.ι.app j) ((fun t => { f := fun i  …
    -/
    apply (hs i).fac
    /-
      🎉 no goals
    -/
  uniq t m hm := by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom s.pt t.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) m) (t.ι.app …
      ⊢ Eq m ((fun t => { f := fun i => (hs i).desc ((HomologicalComplex.eval C c i) …
    -/
    ext i
    /-
      case h
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom s.pt t.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) m) (t.ι.app …
      i : ι
      ⊢ Eq (m.f i) (((fun t => { f := fun i => (hs i).desc ((HomologicalComplex.eval …
    -/
    apply (hs i).uniq ((eval C c i).mapCocone t)
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom s.pt t.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) m) (t.ι.app …
      i : ι
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval …
    -/
    intro j
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom s.pt t.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) m) (t.ι.app …
      i : ι
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.eval C c i).map …
    -/
    dsimp
    /-
      case h.x
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝² : CategoryTheory.Category.{?u.37210, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.37214, u_3} J
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      F : CategoryTheory.Functor J (HomologicalComplex C c)
      s : CategoryTheory.Limits.Cocone F
      hs : (i : ι) → CategoryTheory.Limits.IsColimit ((HomologicalComplex.eval C c i …
      t : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom s.pt t.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) m) (t.ι.app …
      i : ι
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.ι.app j).f i) (m.f i)) ((t.ι.app  …
    -/
    simp only [← comp_f, hm]
    /-
      🎉 no goals
    -/



/-- A cocone for a functor `F : J ⥤ HomologicalComplex C c` which is given in degree `n` by
the colimit of `F ⋙ eval C c n`. -/
@[simps]
noncomputable def coconeOfHasColimitEval : Cocone F where
  pt :=
    { X := fun n => colimit (F ⋙ eval C c n)
      d := fun n m => colimMap { app := fun j => (F.obj j).d n m }
      shape := fun {n m} h => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          n m : ι
          h : Not (c.Rel n m)
          ⊢ Eq ((fun n m => CategoryTheory.Limits.colimMap { app := fun j => (F.obj j).d …
        -/
        ext j
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
        -/
        rw [ι_colimMap]
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ app := fun j => (F.obj j).d n m, n …
        -/
        dsimp
        /-
          case w
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          n m : ι
          h : Not (c.Rel n m)
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).d n m) (CategoryTheory.Lim …
        -/
        rw [(F.obj j).shape _ _ h, zero_comp, comp_zero] }
        /-
          🎉 no goals
        -/
  ι :=
    { app := fun j => { f := fun n => colimit.ι (F ⋙ eval C c n) j }
      naturality := fun i j φ => by
        /-
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          i j : J
          φ : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun j => { f := fun n =>  …
        -/
        ext n
        /-
          case h
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          i j : J
          φ : Quiver.Hom i j
          n : ι
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map φ) ((fun j => { f := fun n => …
        -/
        dsimp
        /-
          case h
          C : Type u_1
          ι : Type u_2
          J : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.42974, u_1} C
          inst✝² : CategoryTheory.Category.{?u.42978, u_3} J
          c : ComplexShape ι
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          F : CategoryTheory.Functor J (HomologicalComplex C c)
          inst✝ : ∀ (n : ι), CategoryTheory.Limits.HasColimit (F.comp (HomologicalComple …
          i j : J
          φ : Quiver.Hom i j
          n : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map φ).f n) (CategoryTheory.Limit …
        -/
        erw [colimit.w (F ⋙ eval C c n) φ, comp_id] }
        /-
          🎉 no goals
        -/


/-- The cocone `coconeOfHasLimitEval F` is colimit. -/
noncomputable def isColimitCoconeOfHasColimitEval : IsColimit (coconeOfHasColimitEval F) :=
  isColimitOfEval _ _ (fun _ => colimit.isColimit _)


instance : HasColimit F := ⟨⟨⟨_, isColimitCoconeOfHasColimitEval F⟩⟩⟩


noncomputable instance (n : ι) : PreservesColimit F (eval C c n) :=
  preservesColimit_of_preserves_colimit_cocone (isColimitCoconeOfHasColimitEval F)
    (colimit.isColimit _)


instance [HasColimitsOfShape J C] : HasColimitsOfShape J (HomologicalComplex C c) := ⟨inferInstance⟩


noncomputable instance [HasColimitsOfShape J C] (n : ι) :
  PreservesColimitsOfShape J (eval C c n) := ⟨inferInstance⟩


instance [HasFiniteColimits C] : HasFiniteColimits (HomologicalComplex C c) :=
  ⟨fun _ _ => inferInstance⟩


noncomputable instance [HasFiniteColimits C] (n : ι) :
  PreservesFiniteColimits (eval C c n) := ⟨fun _ _ _ => inferInstance⟩


instance [HasFiniteColimits C] {K L : HomologicalComplex C c} (φ : K ⟶ L) [Epi φ] (n : ι) :
    Epi (φ.f n) := by
  /-
    C : Type u_1
    ι : Type u_2
    J : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.72857, u_3} J
    c : ComplexShape ι
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    inst✝ : CategoryTheory.Epi φ
    n : ι
    ⊢ CategoryTheory.Epi (φ.f n)
  -/
  change Epi ((HomologicalComplex.eval C c n).map φ)
  /-
    C : Type u_1
    ι : Type u_2
    J : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.72857, u_3} J
    c : ComplexShape ι
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    inst✝ : CategoryTheory.Epi φ
    n : ι
    ⊢ CategoryTheory.Epi ((HomologicalComplex.eval C c n).map φ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A functor `D ⥤ HomologicalComplex C c` preserves limits of shape `J`
if for any `i`, `G ⋙ eval C c i` does. -/
lemma preservesLimitsOfShape_of_eval {D : Type*} [Category D]
    (G : D ⥤ HomologicalComplex C c)
    (_ : ∀ (i : ι), PreservesLimitsOfShape J (G ⋙ eval C c i)) :
    PreservesLimitsOfShape J G :=
  ⟨fun {_} => ⟨fun hs ↦ ⟨isLimitOfEval _ _
    (fun i => isLimitOfPreserves (G ⋙ eval C c i) hs)⟩⟩⟩


/-- A functor `D ⥤ HomologicalComplex C c` preserves colimits of shape `J`
if for any `i`, `G ⋙ eval C c i` does. -/
lemma preservesColimitsOfShape_of_eval {D : Type*} [Category D]
    (G : D ⥤ HomologicalComplex C c)
    (_ : ∀ (i : ι), PreservesColimitsOfShape J (G ⋙ eval C c i)) :
    PreservesColimitsOfShape J G :=
  ⟨fun {_} => ⟨fun hs ↦ ⟨isColimitOfEval _ _
    (fun i => isColimitOfPreserves (G ⋙ eval C c i) hs)⟩⟩⟩


noncomputable instance : PreservesLimitsOfShape J (single C c i) :=
  preservesLimitsOfShape_of_eval _ (fun j => by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_3} J
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : DecidableEq ι
      i j : ι
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((HomologicalComplex.single C …
    -/
    by_cases h : j = i
      /-
        case pos
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        i j : ι
        h : Eq j i
        ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((HomologicalComplex.single C …
      -/
    · subst h
      /-
        case pos
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        j : ι
        ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((HomologicalComplex.single C …
      -/
      exact preservesLimitsOfShape_of_natIso (singleCompEvalIsoSelf C c j).symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        i j : ι
        h : Not (Eq j i)
        ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((HomologicalComplex.single C …
      -/
    · exact Functor.preservesLimitsOfShape_of_isZero _ (isZero_single_comp_eval C c _ _ h) _)
      /-
        🎉 no goals
      -/


noncomputable instance : PreservesColimitsOfShape J (single C c i) :=
  preservesColimitsOfShape_of_eval _ (fun j => by
    /-
      C : Type u_1
      ι : Type u_2
      J : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_3} J
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : DecidableEq ι
      i j : ι
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((HomologicalComplex.single …
    -/
    by_cases h : j = i
      /-
        case pos
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        i j : ι
        h : Eq j i
        ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((HomologicalComplex.single …
      -/
    · subst h
      /-
        case pos
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        j : ι
        ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((HomologicalComplex.single …
      -/
      exact preservesColimitsOfShape_of_natIso (singleCompEvalIsoSelf C c j).symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        ι : Type u_2
        J : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
        inst✝³ : CategoryTheory.Category.{u_4, u_3} J
        c : ComplexShape ι
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : DecidableEq ι
        i j : ι
        h : Not (Eq j i)
        ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((HomologicalComplex.single …
      -/
    · exact Functor.preservesColimitsOfShape_of_isZero _ (isZero_single_comp_eval C c _ _ h) _)
      /-
        🎉 no goals
      -/


                                                                     /-
                                                                       C : Type u_1
                                                                       ι : Type u_2
                                                                       J : Type u_3
                                                                       inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
                                                                       inst✝³ : CategoryTheory.Category.{?u.84859, u_3} J
                                                                       c : ComplexShape ι
                                                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                       inst✝ : DecidableEq ι
                                                                       i : ι
                                                                       ⊢ ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryTheor …
                                                                     -/
noncomputable instance : PreservesFiniteLimits (single C c i) := ⟨by intros; infer_instance⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                                       /-
                                                                         C : Type u_1
                                                                         ι : Type u_2
                                                                         J : Type u_3
                                                                         inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
                                                                         inst✝³ : CategoryTheory.Category.{?u.85766, u_3} J
                                                                         c : ComplexShape ι
                                                                         inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                         inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                         inst✝ : DecidableEq ι
                                                                         i : ι
                                                                         ⊢ ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryTheor …
                                                                       -/
noncomputable instance : PreservesFiniteColimits (single C c i) := ⟨by intros; infer_instance⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


