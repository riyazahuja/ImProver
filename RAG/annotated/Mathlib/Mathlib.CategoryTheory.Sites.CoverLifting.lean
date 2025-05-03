/-- A functor `G : (C, J) ⥤ (D, K)` between sites is called cocontinuous (SGA 4 III 2.1)
if for all covering sieves `R` in `D`, `R.pullback G` is a covering sieve in `C`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`
class Functor.IsCocontinuous : Prop where
  cover_lift : ∀ {U : C} {S : Sieve (G.obj U)} (_ : S ∈ K (G.obj U)), S.functorPullback G ∈ J U


lemma Functor.cover_lift [G.IsCocontinuous J K] {U : C} {S : Sieve (G.obj U)}
    (hS : S ∈ K (G.obj U)) : S.functorPullback G ∈ J U :=
  IsCocontinuous.cover_lift hS


/-- The identity functor on a site is cocontinuous. -/
instance isCocontinuous_id : Functor.IsCocontinuous (𝟭 C) J J :=
               /-
                 C : Type u_1
                 inst✝² : CategoryTheory.Category.{u_4, u_1} C
                 D : Type u_2
                 inst✝¹ : CategoryTheory.Category.{?u.1879, u_2} D
                 E : Type u_3
                 inst✝ : CategoryTheory.Category.{?u.1886, u_3} E
                 G : CategoryTheory.Functor C D
                 G' : CategoryTheory.Functor D E
                 J : CategoryTheory.GrothendieckTopology C
                 K : CategoryTheory.GrothendieckTopology D
                 L : CategoryTheory.GrothendieckTopology E
                 U✝ : C
                 S✝ : CategoryTheory.Sieve ((CategoryTheory.Functor.id C).obj U✝)
                 h : Membership.mem (J ((CategoryTheory.Functor.id C).obj U✝)) S✝
                 ⊢ Membership.mem (J U✝) (CategoryTheory.Sieve.functorPullback (CategoryTheory. …
               -/
  ⟨fun h => by simpa using h⟩
               /-
                 🎉 no goals
               -/


/-- The composition of two cocontinuous functors is cocontinuous. -/
theorem isCocontinuous_comp [G.IsCocontinuous J K] [G'.IsCocontinuous K L] :
    (G ⋙ G').IsCocontinuous J L where
  cover_lift h := G.cover_lift J K (G'.cover_lift K L h)


/-- Auxiliary definition for `lift`. -/
def liftAux {Y : C} (f : G.obj Y ⟶ X) : s.pt ⟶ F.obj (op Y) :=
  Multifork.IsLimit.lift (hF.isLimitMultifork ⟨_, G.cover_lift J K (K.pullback_stable f S.2)⟩)
    (fun k ↦ s.ι (⟨_, G.map k.f ≫ f, k.hf⟩) ≫ α.app (op k.Y)) (by
      /-
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{?u.7617, u_1} C
        inst✝² : CategoryTheory.Category.{?u.7621, u_2} D
        G : CategoryTheory.Functor C D
        A : Type w
        inst✝¹ : CategoryTheory.Category.{w', w} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        inst✝ : G.IsCocontinuous J K
        F : CategoryTheory.Functor (Opposite C) A
        hF : CategoryTheory.Presheaf.IsSheaf J F
        R : CategoryTheory.Functor (Opposite D) A
        α : Quiver.Hom (G.op.comp R) F
        hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
        X : D
        S : K.Cover X
        s : CategoryTheory.Limits.Multifork (S.index R)
        Y : C
        f : Quiver.Hom (G.obj Y) X
        ⊢ ∀ (b : (CategoryTheory.GrothendieckTopology.Cover.index ⟨CategoryTheory.Siev …
      -/
      rintro ⟨⟨Y₁, p₁, hp₁⟩, ⟨Y₂, p₂, hp₂⟩, W, g₁, g₂, w⟩
      /-
        case mk.mk.mk.mk
        C : Type u_1
        D : Type u_2
        inst✝³ : CategoryTheory.Category.{?u.7617, u_1} C
        inst✝² : CategoryTheory.Category.{?u.7621, u_2} D
        G : CategoryTheory.Functor C D
        A : Type w
        inst✝¹ : CategoryTheory.Category.{w', w} A
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        inst✝ : G.IsCocontinuous J K
        F : CategoryTheory.Functor (Opposite C) A
        hF : CategoryTheory.Presheaf.IsSheaf J F
        R : CategoryTheory.Functor (Opposite D) A
        α : Quiver.Hom (G.op.comp R) F
        hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
        X : D
        S : K.Cover X
        s : CategoryTheory.Limits.Multifork (S.index R)
        Y : C
        f : Quiver.Hom (G.obj Y) X
        Y₁ : C
        p₁ : Quiver.Hom Y₁ Y
        hp₁ : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback …
        Y₂ : C
        p₂ : Quiver.Hom Y₂ Y
        hp₂ : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback …
        W : C
        g₁ : Quiver.Hom W { Y := Y₁, f := p₁, hf := hp₁ }.Y
        g₂ : Quiver.Hom W { Y := Y₂, f := p₂, hf := hp₂ }.Y
        w : Eq (CategoryTheory.CategoryStruct.comp g₁ { Y := Y₁, f := p₁, hf := hp₁ }. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun k => CategoryTheory.CategoryStr …
      -/
      dsimp at g₁ g₂ w ⊢
      simp only [Category.assoc, ← α.naturality, Functor.comp_map,
        Functor.op_map, Quiver.Hom.unop_op]
      apply s.condition_assoc
        (GrothendieckTopology.Cover.Relation.mk
          { hf := hp₁ }
          { hf := hp₂ }
          { g₁ := G.map g₁
            g₂ := G.map g₂
            w := by simpa using G.congr_map w =≫ f }))


lemma liftAux_map {Y : C} (f : G.obj Y ⟶ X) {W : C} (g : W ⟶ Y) (i : S.Arrow)
    (h : G.obj W ⟶ i.Y) (w : h ≫ i.f = G.map g ≫ f) :
    liftAux hF α s f ≫ F.map g.op = s.ι i ≫ R.map h.op ≫ α.app _ :=
  (Multifork.IsLimit.fac
    (hF.isLimitMultifork ⟨_, G.cover_lift J K (K.pullback_stable f S.2)⟩) _ _
      ⟨W, g, by simpa only [Sieve.functorPullback_apply, functorPullback_mem,
        Sieve.pullback_apply, ← w] using S.1.downward_closed i.hf h⟩).trans (by
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{u_4, u_1} C
          inst✝² : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          A : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} A
          J : CategoryTheory.GrothendieckTopology C
          K : CategoryTheory.GrothendieckTopology D
          inst✝ : G.IsCocontinuous J K
          F : CategoryTheory.Functor (Opposite C) A
          hF : CategoryTheory.Presheaf.IsSheaf J F
          R : CategoryTheory.Functor (Opposite D) A
          α : Quiver.Hom (G.op.comp R) F
          X : D
          S : K.Cover X
          s : CategoryTheory.Limits.Multifork (S.index R)
          Y : C
          f : Quiver.Hom (G.obj Y) X
          W : C
          g : Quiver.Hom W Y
          i : S.Arrow
          h : Quiver.Hom (G.obj W) i.Y
          w : Eq (CategoryTheory.CategoryStruct.comp h i.f) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι { Y := G.obj { Y := W, f := g, h …
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{u_4, u_1} C
          inst✝² : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          A : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} A
          J : CategoryTheory.GrothendieckTopology C
          K : CategoryTheory.GrothendieckTopology D
          inst✝ : G.IsCocontinuous J K
          F : CategoryTheory.Functor (Opposite C) A
          hF : CategoryTheory.Presheaf.IsSheaf J F
          R : CategoryTheory.Functor (Opposite D) A
          α : Quiver.Hom (G.op.comp R) F
          X : D
          S : K.Cover X
          s : CategoryTheory.Limits.Multifork (S.index R)
          Y : C
          f : Quiver.Hom (G.obj Y) X
          W : C
          g : Quiver.Hom W Y
          i : S.Arrow
          h : Quiver.Hom (G.obj W) i.Y
          w : Eq (CategoryTheory.CategoryStruct.comp h i.f) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι { Y := G.obj W, f := CategoryThe …
        -/
        simp only [← Category.assoc]
        /-
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{u_4, u_1} C
          inst✝² : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          A : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} A
          J : CategoryTheory.GrothendieckTopology C
          K : CategoryTheory.GrothendieckTopology D
          inst✝ : G.IsCocontinuous J K
          F : CategoryTheory.Functor (Opposite C) A
          hF : CategoryTheory.Presheaf.IsSheaf J F
          R : CategoryTheory.Functor (Opposite D) A
          α : Quiver.Hom (G.op.comp R) F
          X : D
          S : K.Cover X
          s : CategoryTheory.Limits.Multifork (S.index R)
          Y : C
          f : Quiver.Hom (G.obj Y) X
          W : C
          g : Quiver.Hom W Y
          i : S.Arrow
          h : Quiver.Hom (G.obj W) i.Y
          w : Eq (CategoryTheory.CategoryStruct.comp h i.f) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι { Y := G.obj W, f := CategoryThe …
        -/
        congr 1
        let r : S.Relation :=
          GrothendieckTopology.Cover.Relation.mk
            { f := G.map g ≫ f
              hf := by simpa only [← w] using S.1.downward_closed i.hf h } i
            { g₁ := 𝟙 _
              g₂ := h
              w := by simpa using w.symm }
        /-
          case e_a
          C : Type u_1
          D : Type u_2
          inst✝³ : CategoryTheory.Category.{u_4, u_1} C
          inst✝² : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          A : Type w
          inst✝¹ : CategoryTheory.Category.{w', w} A
          J : CategoryTheory.GrothendieckTopology C
          K : CategoryTheory.GrothendieckTopology D
          inst✝ : G.IsCocontinuous J K
          F : CategoryTheory.Functor (Opposite C) A
          hF : CategoryTheory.Presheaf.IsSheaf J F
          R : CategoryTheory.Functor (Opposite D) A
          α : Quiver.Hom (G.op.comp R) F
          X : D
          S : K.Cover X
          s : CategoryTheory.Limits.Multifork (S.index R)
          Y : C
          f : Quiver.Hom (G.obj Y) X
          W : C
          g : Quiver.Hom W Y
          i : S.Arrow
          h : Quiver.Hom (G.obj W) i.Y
          w : Eq (CategoryTheory.CategoryStruct.comp h i.f) (CategoryTheory.CategoryStru …
          r : S.Relation := { fst := { Y := G.obj W, f := CategoryTheory.CategoryStruct. …
          ⊢ Eq (s.ι { Y := G.obj W, f := CategoryTheory.CategoryStruct.comp (G.map g) f, …
        -/
        simpa [r] using s.condition r )
        /-
          🎉 no goals
        -/


lemma liftAux_map' {Y Y' : C} (f : G.obj Y ⟶ X) (f' : G.obj Y' ⟶ X) {W : C}
    (a : W ⟶ Y) (b : W ⟶ Y') (w : G.map a ≫ f = G.map b ≫ f') :
    liftAux hF α s f ≫ F.map a.op = liftAux hF α s f' ≫ F.map b.op := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoconti …
  -/
  apply hF.hom_ext ⟨_, G.cover_lift J K (K.pullback_stable (G.map a ≫ f) S.2)⟩
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    ⊢ ∀ (I : CategoryTheory.GrothendieckTopology.Cover.Arrow ⟨CategoryTheory.Sieve …
  -/
  rintro ⟨T, g, hg⟩
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  have eq₁ := liftAux_map hF α s f (g ≫ a) ⟨_, _, hg⟩ (𝟙 _) (by simp)
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  have eq₂ := liftAux_map hF α s f' (g ≫ b) ⟨_, _, hg⟩ (𝟙 _) (by simp [w])
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp at eq₁ eq₂
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Functor.map_comp, Functor.map_id, Category.id_comp] at eq₁ eq₂
  /-
    case h.mk
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    Y Y' : C
    f : Quiver.Hom (G.obj Y) X
    f' : Quiver.Hom (G.obj Y') X
    W : C
    a : Quiver.Hom W Y
    b : Quiver.Hom W Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (G.map a) f) (CategoryTheory.Catego …
    T : C
    g : Quiver.Hom T W
    hg : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, eq₁, eq₂]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `isLimitMultifork` -/
def lift : s.pt ⟶ R.obj (op X) :=
  (hR (op X)).lift (Cone.mk _
    { app := fun j ↦ liftAux hF α s j.hom.unop
      naturality := fun j j' φ ↦ by
        simpa using liftAux_map' hF α s j'.hom.unop j.hom.unop (𝟙 _) φ.right.unop
          (Quiver.Hom.op_inj (by simpa using (StructuredArrow.w φ).symm)) })


lemma fac' (j : StructuredArrow (op X) G.op) :
    lift hF hR s ≫ R.map j.hom ≫ α.app j.right = liftAux hF α s j.hom.unop := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoconti …
  -/
  apply IsLimit.fac
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma fac (i : S.Arrow) : lift hF hR s ≫ R.map i.f.op = s.ι i := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoconti …
  -/
  apply (hR (op i.Y)).hom_ext
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    ⊢ ∀ (j : CategoryTheory.StructuredArrow { unop := i.Y } G.op), Eq (CategoryThe …
  -/
  intro j
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    j : CategoryTheory.StructuredArrow { unop := i.Y } G.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  have eq := fac' hF hR s (StructuredArrow.mk (i.f.op ≫ j.hom))
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    j : CategoryTheory.StructuredArrow { unop := i.Y } G.op
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoco …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp at eq ⊢
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    j : CategoryTheory.StructuredArrow { unop := i.Y } G.op
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoco …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Functor.map_comp, Category.assoc] at eq
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    j : CategoryTheory.StructuredArrow { unop := i.Y } G.op
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoco …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, eq]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    s : CategoryTheory.Limits.Multifork (S.index R)
    i : S.Arrow
    j : CategoryTheory.StructuredArrow { unop := i.Y } G.op
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.RanIsSheafOfIsCoco …
    ⊢ Eq (CategoryTheory.RanIsSheafOfIsCocontinuous.liftAux hF α s (CategoryTheory …
  -/
  simpa using liftAux_map hF α s (j.hom.unop ≫ i.f) (𝟙 _) i j.hom.unop (by simp)
  /-
    🎉 no goals
  -/


include hR hF in
variable (K) in
lemma hom_ext {W : A} {f g : W ⟶ R.obj (op X)}
    (h : ∀ (i : S.Arrow), f ≫ R.map i.f.op = g ≫ R.map i.f.op) : f = g := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W : A
    f g : Quiver.Hom W (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    ⊢ Eq f g
  -/
  apply (hR (op X)).hom_ext
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W : A
    f g : Quiver.Hom W (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    ⊢ ∀ (j : CategoryTheory.StructuredArrow { unop := X } G.op), Eq (CategoryTheor …
  -/
  intro j
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W : A
    f g : Quiver.Hom W (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (((CategoryTheory.Functor.RightExte …
  -/
  apply hF.hom_ext ⟨_, G.cover_lift J K (K.pullback_stable j.hom.unop S.2)⟩
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W : A
    f g : Quiver.Hom W (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    ⊢ ∀ (I : CategoryTheory.GrothendieckTopology.Cover.Arrow ⟨CategoryTheory.Sieve …
  -/
  intro ⟨W, i, hi⟩
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W✝ : A
    f g : Quiver.Hom W✝ (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    W : C
    i : Quiver.Hom W (Opposite.unop j.right)
    hi : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  have eq := h (GrothendieckTopology.Cover.Arrow.mk _ (G.map i ≫ j.hom.unop) hi)
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W✝ : A
    f g : Quiver.Hom W✝ (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    W : C
    i : Quiver.Hom W (Opposite.unop j.right)
    hi : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq : Eq (CategoryTheory.CategoryStruct.comp f (R.map { Y := G.obj W, f := Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  dsimp at eq ⊢
  simp only [Category.assoc, ← NatTrans.naturality, Functor.comp_map, ← Functor.map_comp_assoc,
    Functor.op_map, Quiver.Hom.unop_op]
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.43894, u_1} C
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝¹ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCocontinuous J K
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf J F
    R : CategoryTheory.Functor (Opposite D) A
    α : Quiver.Hom (G.op.comp R) F
    hR : (CategoryTheory.Functor.RightExtension.mk R α).IsPointwiseRightKanExtension
    X : D
    S : K.Cover X
    W✝ : A
    f g : Quiver.Hom W✝ (R.obj { unop := X })
    h : ∀ (i : S.Arrow), Eq (CategoryTheory.CategoryStruct.comp f (R.map i.f.op))  …
    j : CategoryTheory.StructuredArrow { unop := X } G.op
    W : C
    i : Quiver.Hom W (Opposite.unop j.right)
    hi : (↑⟨CategoryTheory.Sieve.functorPullback G (CategoryTheory.Sieve.pullback  …
    eq : Eq (CategoryTheory.CategoryStruct.comp f (R.map (CategoryTheory.CategoryS …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
  -/
  rw [reassoc_of% eq]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `ran_isSheaf_of_isCocontinuous`: if `G : C ⥤ D` is a
cocontinuous functor,   -/
def isLimitMultifork : IsLimit (S.multifork R) :=
  Multifork.IsLimit.mk _ (lift hF hR) (fac hF hR)
    (fun s _ hm ↦ hom_ext K hF hR (fun i ↦ (hm i).trans (fac hF hR s i).symm))


/-- If `G` is cocontinuous, then `G.op.ran` pushes sheaves to sheaves.

This is SGA 4 III 2.2. An alternative reference is
https://stacks.math.columbia.edu/tag/00XK (where results
are obtained under the additional assumption that
`C` and `D` have pullbacks).
-/
theorem ran_isSheaf_of_isCocontinuous (ℱ : Sheaf J A) :
    Presheaf.IsSheaf K (G.op.ran.obj ℱ.val) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝¹ : G.IsCocontinuous J K
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRightK …
    ℱ : CategoryTheory.Sheaf J A
    ⊢ CategoryTheory.Presheaf.IsSheaf K (G.op.ran.obj ℱ.val)
  -/
  rw [Presheaf.isSheaf_iff_multifork]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝² : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝¹ : G.IsCocontinuous J K
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRightK …
    ℱ : CategoryTheory.Sheaf J A
    ⊢ ∀ (X : D) (S : K.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.multif …
  -/
  intros X S
  exact ⟨RanIsSheafOfIsCocontinuous.isLimitMultifork ℱ.2
    (G.op.isPointwiseRightKanExtensionRanCounit ℱ.val) S⟩


/-- A cocontinuous functor induces a pushforward functor on categories of sheaves. -/
def Functor.sheafPushforwardCocontinuous : Sheaf J A ⥤ Sheaf K A where
  obj ℱ := ⟨G.op.ran.obj ℱ.val, ran_isSheaf_of_isCocontinuous _ K ℱ⟩
  map f := ⟨G.op.ran.map f.val⟩
  map_id ℱ := Sheaf.Hom.ext <| (ran G.op).map_id ℱ.val
  map_comp f g := Sheaf.Hom.ext <| (ran G.op).map_comp f.val g.val


/-- `G.sheafPushforwardCocontinuous A J K : Sheaf J A ⥤ Sheaf K A` is induced
by the right Kan extension functor `G.op.ran` on presheaves. -/
@[simps! hom inv]
def Functor.sheafPushforwardCocontinuousCompSheafToPresheafIso :
    G.sheafPushforwardCocontinuous A J K ⋙ sheafToPresheaf K A ≅
      sheafToPresheaf J A ⋙ G.op.ran := Iso.refl _

/-

Given a cocontinuous functor `G`, the precomposition with `G.op` induces a functor
on presheaves with leads to a "pullback" functor `Sheaf K A ⥤ Sheaf J A` (TODO: formalize
this as `G.sheafPullbackCocontinuous A J K`) using the associated sheaf functor.
It is shown in SGA 4 III 2.3 that this pullback functor is
left adjoint to `G.sheafPushforwardCocontinuous A J K`. This adjunction may replace
`Functor.sheafAdjunctionCocontinuous` below, and then, it could be shown that if
`G` is also continuous, then we have an isomorphism
`G.sheafPullbackCocontinuous A J K ≅ G.sheafPushforwardContinuous A J K` (TODO).

-/


/--
Given a functor between sites that is continuous and cocontinuous,
the pushforward for the continuous functor `G` is left adjoint to
the pushforward for the cocontinuous functor `G`. -/
noncomputable def sheafAdjunctionCocontinuous :
    G.sheafPushforwardContinuous A J K ⊣ G.sheafPushforwardCocontinuous A J K :=
  (G.op.ranAdjunction A).restrictFullyFaithful
    (fullyFaithfulSheafToPresheaf K A) (fullyFaithfulSheafToPresheaf J A)
    (G.sheafPushforwardContinuousCompSheafToPresheafIso A J K).symm
    (G.sheafPushforwardCocontinuousCompSheafToPresheafIso A J K).symm


lemma sheafAdjunctionCocontinuous_unit_app_val (F : Sheaf K A) :
    ((G.sheafAdjunctionCocontinuous A J K).unit.app F).val =
      (G.op.ranAdjunction A).unit.app F.val := by
  apply ((G.op.ranAdjunction A).map_restrictFullyFaithful_unit_app
    (fullyFaithfulSheafToPresheaf K A) (fullyFaithfulSheafToPresheaf J A)
    (G.sheafPushforwardContinuousCompSheafToPresheafIso A J K).symm
    (G.sheafPushforwardCocontinuousCompSheafToPresheafIso A J K).symm F).trans
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝³ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝² : G.IsCocontinuous J K
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝ : G.IsContinuous J K
    F : CategoryTheory.Sheaf K A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.op.ranAdjunction A).unit.app ((Ca …
  -/
  dsimp
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝³ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝² : G.IsCocontinuous J K
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝ : G.IsContinuous J K
    F : CategoryTheory.Sheaf K A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.op.ranAdjunction A).unit.app F.va …
  -/
  erw [Functor.map_id]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝³ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝² : G.IsCocontinuous J K
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝ : G.IsContinuous J K
    F : CategoryTheory.Sheaf K A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.op.ranAdjunction A).unit.app F.va …
  -/
  change _ ≫ 𝟙 _ ≫ 𝟙 _ = _
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝³ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝² : G.IsCocontinuous J K
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝ : G.IsContinuous J K
    F : CategoryTheory.Sheaf K A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.op.ranAdjunction A).unit.app F.va …
  -/
  simp only [Category.comp_id]
  /-
    🎉 no goals
  -/


lemma sheafAdjunctionCocontinuous_counit_app_val (F : Sheaf J A) :
    ((G.sheafAdjunctionCocontinuous A J K).counit.app F).val =
      (G.op.ranAdjunction A).counit.app F.val :=
  ((G.op.ranAdjunction A).map_restrictFullyFaithful_counit_app
    (fullyFaithfulSheafToPresheaf K A) (fullyFaithfulSheafToPresheaf J A)
    (G.sheafPushforwardContinuousCompSheafToPresheafIso A J K).symm
    (G.sheafPushforwardCocontinuousCompSheafToPresheafIso A J K).symm F).trans
          /-
            C : Type u_1
            D : Type u_2
            inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
            inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
            G : CategoryTheory.Functor C D
            A : Type w
            inst✝³ : CategoryTheory.Category.{w', w} A
            J : CategoryTheory.GrothendieckTopology C
            K : CategoryTheory.GrothendieckTopology D
            inst✝² : G.IsCocontinuous J K
            inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
            inst✝ : G.IsContinuous J K
            F : CategoryTheory.Sheaf J A
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.sheafPushforwardContinuousCompShe …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/


lemma sheafAdjunctionCocontinuous_homEquiv_apply_val {F : Sheaf K A} {H : Sheaf J A}
    (f : (G.sheafPushforwardContinuous A J K).obj F ⟶ H) :
    ((G.sheafAdjunctionCocontinuous A J K).homEquiv F H f).val =
      (G.op.ranAdjunction A).homEquiv F.val H.val f.val :=
  ((sheafToPresheaf K A).congr_map
    (((G.op.ranAdjunction A).restrictFullyFaithful_homEquiv_apply
      (fullyFaithfulSheafToPresheaf K A) (fullyFaithfulSheafToPresheaf J A)
      (G.sheafPushforwardContinuousCompSheafToPresheafIso A J K).symm
      (G.sheafPushforwardCocontinuousCompSheafToPresheafIso A J K).symm f))).trans (by
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
          inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
          G : CategoryTheory.Functor C D
          A : Type w
          inst✝³ : CategoryTheory.Category.{w', w} A
          J : CategoryTheory.GrothendieckTopology C
          K : CategoryTheory.GrothendieckTopology D
          inst✝² : G.IsCocontinuous J K
          inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
          inst✝ : G.IsContinuous J K
          F : CategoryTheory.Sheaf K A
          H : CategoryTheory.Sheaf J A
          f : Quiver.Hom ((G.sheafPushforwardContinuous A J K).obj F) H
          ⊢ Eq ((CategoryTheory.sheafToPresheaf K A).map ((CategoryTheory.fullyFaithfulS …
        -/
        dsimp
        erw [Functor.map_id, Category.comp_id, Category.id_comp,
          Adjunction.homEquiv_unit])


/-- The natural isomorphism exhibiting compatibility between pushforward and sheafification. -/
def pushforwardContinuousSheafificationCompatibility [G.IsContinuous J K] :
    (whiskeringLeft _ _ A).obj G.op ⋙ presheafToSheaf J A ≅
    presheafToSheaf K A ⋙ G.sheafPushforwardContinuous A J K :=
  ((G.op.ranAdjunction A).comp (sheafificationAdjunction J A)).leftAdjointUniq
    ((sheafificationAdjunction K A).comp (G.sheafAdjunctionCocontinuous A J K))

/- Implementation: This is primarily used to prove the lemma
`pullbackSheafificationCompatibility_hom_app_val`. -/

lemma toSheafify_pullbackSheafificationCompatibility (F : Dᵒᵖ ⥤ A) :
    toSheafify J (G.op ⋙ F) ≫
    ((G.pushforwardContinuousSheafificationCompatibility A J K).hom.app F).val =
    whiskerLeft _ (toSheafify K _) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (G.op.co …
  -/
  let adj₁ := G.op.ranAdjunction A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (G.op.co …
  -/
  let adj₂ := sheafificationAdjunction J A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (G.op.co …
  -/
  let adj₃ := sheafificationAdjunction K A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (G.op.co …
  -/
  let adj₄ := G.sheafAdjunctionCocontinuous A J K
  change adj₂.unit.app (((whiskeringLeft Cᵒᵖ Dᵒᵖ A).obj G.op).obj F) ≫
    (sheafToPresheaf J A).map (((adj₁.comp adj₂).leftAdjointUniq (adj₃.comp adj₄)).hom.app F) =
      ((whiskeringLeft Cᵒᵖ Dᵒᵖ A).obj G.op).map (adj₃.unit.app F)
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (((CategoryTheory.whis …
  -/
  apply (adj₁.homEquiv _ _).injective
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    ⊢ Eq ((adj₁.homEquiv F ((CategoryTheory.sheafToPresheaf J A).obj (((CategoryTh …
  -/
  have eq := (adj₁.comp adj₂).unit_leftAdjointUniq_hom_app (adj₃.comp adj₄) F
  rw [Adjunction.comp_unit_app, Adjunction.comp_unit_app, comp_map,
    Category.assoc] at eq
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    eq : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app F) (CategoryTheory. …
    ⊢ Eq ((adj₁.homEquiv F ((CategoryTheory.sheafToPresheaf J A).obj (((CategoryTh …
  -/
  rw [adj₁.homEquiv_unit, Functor.map_comp, eq]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    eq : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app F) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app F) ((CategoryTheory.sh …
  -/
  apply (adj₁.homEquiv _ _).symm.injective
  simp only [Adjunction.homEquiv_counit, map_comp, Category.assoc,
    Adjunction.homEquiv_unit, Adjunction.unit_naturality]
  /-
    case a.a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    eq : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app F) (CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft (Opp …
  -/
  congr 3
  /-
    case a.a.e_a.e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    adj₁ : CategoryTheory.Adjunction ((CategoryTheory.whiskeringLeft (Opposite C)  …
    adj₂ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf J A) (Categor …
    adj₃ : CategoryTheory.Adjunction (CategoryTheory.presheafToSheaf K A) (Categor …
    adj₄ : CategoryTheory.Adjunction (G.sheafPushforwardContinuous A J K) (G.sheaf …
    eq : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app F) (CategoryTheory. …
    ⊢ Eq ((CategoryTheory.sheafToPresheaf K A).map (adj₄.unit.app ((CategoryTheory …
  -/
  exact G.sheafAdjunctionCocontinuous_unit_app_val A J K ((presheafToSheaf K A).obj F)
  /-
    🎉 no goals
  -/


@[simp]
lemma pushforwardContinuousSheafificationCompatibility_hom_app_val (F : Dᵒᵖ ⥤ A) :
    ((G.pushforwardContinuousSheafificationCompatibility A J K).hom.app F).val =
    sheafifyLift J (whiskerLeft G.op <| toSheafify K F)
      ((presheafToSheaf K A ⋙ G.sheafPushforwardContinuous A J K).obj F).cond := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    ⊢ Eq ((G.pushforwardContinuousSheafificationCompatibility A J K).hom.app F).va …
  -/
  apply sheafifyLift_unique
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    A : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁴ : G.IsCocontinuous J K
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasPointwiseRight …
    inst✝² : G.IsContinuous J K
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : CategoryTheory.HasWeakSheafify K A
    F : CategoryTheory.Functor (Opposite D) A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.toSheafify J (((Categ …
  -/
  apply toSheafify_pullbackSheafificationCompatibility
  /-
    🎉 no goals
  -/


