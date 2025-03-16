/-- For an adjuntion `f ⊣ u`, `u` is an absolute left Kan extension of the identity along `f`.
The unit of this Kan extension is given by the unit of the adjunction. -/
def Adjunction.isAbsoluteLeftKan {f : a ⟶ b} {u : b ⟶ a} (adj : f ⊣ u) :
    IsAbsKan (.mk u adj.unit) := fun {x} h ↦
  .mk (fun s  ↦ LeftExtension.homMk
    (𝟙 _ ⊗≫ u ◁ s.unit ⊗≫ adj.counit ▷ s.extension ⊗≫ 𝟙 _ : u ≫ h ⟶ s.extension) <|
      calc _
        _ = 𝟙 _ ⊗≫ (adj.unit ▷ _ ≫ _ ◁ s.unit) ⊗≫ f ◁ adj.counit ▷ s.extension ⊗≫ 𝟙 _ := by
          dsimp only [whisker_extension, StructuredArrow.mk_right, whisker_unit,
            StructuredArrow.mk_hom_eq_self]
          /-
            B : Type u
            inst✝ : CategoryTheory.Bicategory B
            a b c : B
            f : Quiver.Hom a b
            u : Quiver.Hom b a
            adj : CategoryTheory.Bicategory.Adjunction f u
            x : B
            h : Quiver.Hom a x
            s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.c …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          bicategory
          /-
            🎉 no goals
          -/
        _ = 𝟙 _ ⊗≫ s.unit ⊗≫ leftZigzag adj.unit adj.counit ▷ s.extension ⊗≫ 𝟙 _ := by
          /-
            B : Type u
            inst✝ : CategoryTheory.Bicategory B
            a b c : B
            f : Quiver.Hom a b
            u : Quiver.Hom b a
            adj : CategoryTheory.Bicategory.Adjunction f u
            x : B
            h : Quiver.Hom a x
            s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.c …
            ⊢ Eq (CategoryTheory.bicategoricalComp (CategoryTheory.CategoryStruct.id (Cate …
          -/
          rw [← whisker_exchange]; bicategory
                                   /-
                                     🎉 no goals
                                   -/
        _ = s.unit := by
          /-
            B : Type u
            inst✝ : CategoryTheory.Bicategory B
            a b c : B
            f : Quiver.Hom a b
            u : Quiver.Hom b a
            adj : CategoryTheory.Bicategory.Adjunction f u
            x : B
            h : Quiver.Hom a x
            s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.c …
            ⊢ Eq (CategoryTheory.bicategoricalComp (CategoryTheory.CategoryStruct.id (Cate …
          -/
          rw [adj.left_triangle]; bicategory) <| by
                                  /-
                                    🎉 no goals
                                  -/
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      u : Quiver.Hom b a
      adj : CategoryTheory.Bicategory.Adjunction f u
      x : B
      h : Quiver.Hom a x
      ⊢ ∀ (s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStr …
    -/
    intro s τ₀
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      u : Quiver.Hom b a
      adj : CategoryTheory.Bicategory.Adjunction f u
      x : B
      h : Quiver.Hom a x
      s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.c …
      τ₀ : Quiver.Hom ((CategoryTheory.Bicategory.LeftExtension.mk u adj.unit).whisk …
      ⊢ Eq τ₀ ((fun s => CategoryTheory.Bicategory.LeftExtension.homMk (CategoryTheo …
    -/
    ext
    /- We need to specify the type of `τ` to use the notation `⊗≫`. -/
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      u : Quiver.Hom b a
      adj : CategoryTheory.Bicategory.Adjunction f u
      x : B
      h : Quiver.Hom a x
      s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.c …
      τ₀ : Quiver.Hom ((CategoryTheory.Bicategory.LeftExtension.mk u adj.unit).whisk …
      ⊢ Eq τ₀.right ((fun s => CategoryTheory.Bicategory.LeftExtension.homMk (Catego …
    -/
    let τ : u ≫ h ⟶ s.extension := τ₀.right
    have hτ : adj.unit ▷ h ⊗≫ f ◁ τ = s.unit := by
      simpa [bicategoricalComp] using LeftExtension.w τ₀
    calc τ
      _ = 𝟙 _ ⊗≫ rightZigzag adj.unit adj.counit ▷ h ⊗≫ τ ⊗≫ 𝟙 _ := by
        rw [adj.right_triangle]; bicategory
      _ = 𝟙 _ ⊗≫ u ◁ adj.unit ▷ h ⊗≫ (adj.counit ▷ _ ≫ _ ◁ τ) ⊗≫ 𝟙 _ := by
        rw [rightZigzag]; bicategory
      _ = 𝟙 _ ⊗≫ u ◁ (adj.unit ▷ h ⊗≫ f ◁ τ) ⊗≫ adj.counit ▷ s.extension ⊗≫ 𝟙 _ := by
        rw [← whisker_exchange]; bicategory
      _ = _ := by
        rw [hτ]; dsimp only [StructuredArrow.homMk_right]


/-- A left Kan extension of the identity along `f` such that `f` commutes with is a right adjoint
to `f`. The unit of this adjoint is given by the unit of the Kan extension. -/
def LeftExtension.IsKan.adjunction {f : a ⟶ b} {t : LeftExtension f (𝟙 a)}
    (H : IsKan t) (H' : IsKan (t.whisker f)) :
      f ⊣ t.extension :=
  let ε : t.extension ≫ f ⟶ 𝟙 b := H'.desc <| .mk _ <| (λ_ f).hom ≫ (ρ_ f).inv
  have Hε : leftZigzag t.unit ε = (λ_ f).hom ≫ (ρ_ f).inv := by
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      t : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.i …
      H : t.IsKan
      H' : (t.whisker f).IsKan
      ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension f) (CategoryThe …
      ⊢ Eq (CategoryTheory.Bicategory.leftZigzag t.unit ε) (CategoryTheory.CategoryS …
    -/
    simpa [leftZigzag, bicategoricalComp] using H'.fac <| .mk _ <| (λ_ f).hom ≫ (ρ_ f).inv
    /-
      🎉 no goals
    -/
  { unit := t.unit
    counit := ε
    left_triangle := Hε
    right_triangle := by
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        t : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.i …
        H : t.IsKan
        H' : (t.whisker f).IsKan
        ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension f) (CategoryThe …
        Hε : Eq (CategoryTheory.Bicategory.leftZigzag t.unit ε) (CategoryTheory.Catego …
        ⊢ Eq (CategoryTheory.Bicategory.rightZigzag t.unit ε) (CategoryTheory.Category …
      -/
      apply (cancel_epi (ρ_ _).inv).mp
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        t : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct.i …
        H : t.IsKan
        H' : (t.whisker f).IsKan
        ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension f) (CategoryThe …
        Hε : Eq (CategoryTheory.Bicategory.leftZigzag t.unit ε) (CategoryTheory.Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.rightUnito …
      -/
      apply H.hom_ext
      calc _
        _ = 𝟙 _ ⊗≫ t.unit ⊗≫ f ◁ rightZigzag t.unit ε ⊗≫ 𝟙 _ := by
          bicategory
        _ = 𝟙 _ ⊗≫ (t.unit ▷ _ ≫ _ ◁ t.unit) ⊗≫ f ◁ ε ▷ t.extension ⊗≫ 𝟙 _ := by
          rw [rightZigzag]; bicategory
        _ = 𝟙 _ ⊗≫ t.unit ⊗≫ (t.unit ▷ f ⊗≫ f ◁ ε) ▷ t.extension ⊗≫ 𝟙 _ := by
          rw [← whisker_exchange]; bicategory
        _ = _ := by
          rw [← leftZigzag, Hε]; bicategory }


/-- For an adjuntion `f ⊣ u`, `u` is a left Kan extension of the identity along `f`.
The unit of this Kan extension is given by the unit of the adjunction. -/
def LeftExtension.IsAbsKan.adjunction {f : a ⟶ b} (t : LeftExtension f (𝟙 a)) (H : IsAbsKan t) :
    f ⊣ t.extension :=
  H.isKan.adjunction (H f)


theorem isLeftAdjoint_TFAE (f : a ⟶ b) :
    List.TFAE [
      IsLeftAdjoint f,
      HasAbsLeftKanExtension f (𝟙 a),
      ∃ _ : HasLeftKanExtension f (𝟙 a), Lan.CommuteWith f (𝟙 a) f] := by
  tfae_have 1 → 2
  | h => IsAbsKan.hasAbsLeftKanExtension (Adjunction.ofIsLeftAdjoint f).isAbsoluteLeftKan
  tfae_have 2 → 3
  | h => ⟨inferInstance, inferInstance⟩
  tfae_have 3 → 1
  | ⟨h, h'⟩ => .mk <| (lanIsKan f (𝟙 a)).adjunction <| Lan.CommuteWith.isKan f (𝟙 a) f
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    f : Quiver.Hom a b
    tfae_1_to_2 : CategoryTheory.Bicategory.IsLeftAdjoint f → CategoryTheory.Bicat …
    tfae_2_to_3 : CategoryTheory.Bicategory.HasAbsLeftKanExtension f (CategoryTheo …
    tfae_3_to_1 : (Exists fun x => CategoryTheory.Bicategory.Lan.CommuteWith f (Ca …
    ⊢ (List.cons (CategoryTheory.Bicategory.IsLeftAdjoint f) (List.cons (CategoryT …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- For an adjuntion `f ⊣ u`, `f` is an absolute left Kan lift of the identity along `u`.
The unit of this Kan lift is given by the unit of the adjunction. -/
def Adjunction.isAbsoluteLeftKanLift {f : a ⟶ b} {u : b ⟶ a} (adj : f ⊣ u) :
    IsAbsKan (.mk f adj.unit) := fun {x} h ↦
  .mk (fun s ↦ LeftLift.homMk
    (𝟙 _ ⊗≫ s.unit ▷ f ⊗≫ s.lift ◁ adj.counit ⊗≫ 𝟙 _ : h ≫ f ⟶ s.lift) <|
      calc _
      _ = 𝟙 _ ⊗≫ (_ ◁ adj.unit ≫ s.unit ▷ _) ⊗≫ s.lift ◁ adj.counit ▷ u ⊗≫ 𝟙 _ := by
        dsimp only [whisker_lift, StructuredArrow.mk_right, whisker_unit,
          StructuredArrow.mk_hom_eq_self]
        /-
          B : Type u
          inst✝ : CategoryTheory.Bicategory B
          a b c : B
          f : Quiver.Hom a b
          u : Quiver.Hom b a
          adj : CategoryTheory.Bicategory.Adjunction f u
          x : B
          h : Quiver.Hom x a
          s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        bicategory
        /-
          🎉 no goals
        -/
      _ = s.unit ⊗≫ s.lift ◁ (rightZigzag adj.unit adj.counit) ⊗≫ 𝟙 _ := by
        /-
          B : Type u
          inst✝ : CategoryTheory.Bicategory B
          a b c : B
          f : Quiver.Hom a b
          u : Quiver.Hom b a
          adj : CategoryTheory.Bicategory.Adjunction f u
          x : B
          h : Quiver.Hom x a
          s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
          ⊢ Eq (CategoryTheory.bicategoricalComp (CategoryTheory.CategoryStruct.id (Cate …
        -/
        rw [whisker_exchange, rightZigzag]; bicategory
                                            /-
                                              🎉 no goals
                                            -/
      _ = s.unit := by
        /-
          B : Type u
          inst✝ : CategoryTheory.Bicategory B
          a b c : B
          f : Quiver.Hom a b
          u : Quiver.Hom b a
          adj : CategoryTheory.Bicategory.Adjunction f u
          x : B
          h : Quiver.Hom x a
          s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
          ⊢ Eq (CategoryTheory.bicategoricalComp s.unit (CategoryTheory.bicategoricalCom …
        -/
        rw [adj.right_triangle]; bicategory) <| by
                                 /-
                                   🎉 no goals
                                 -/
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        u : Quiver.Hom b a
        adj : CategoryTheory.Bicategory.Adjunction f u
        x : B
        h : Quiver.Hom x a
        ⊢ ∀ (s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.c …
      -/
      intro s τ₀
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        u : Quiver.Hom b a
        adj : CategoryTheory.Bicategory.Adjunction f u
        x : B
        h : Quiver.Hom x a
        s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
        τ₀ : Quiver.Hom ((CategoryTheory.Bicategory.LeftLift.mk f adj.unit).whisker h) s
        ⊢ Eq τ₀ ((fun s => CategoryTheory.Bicategory.LeftLift.homMk (CategoryTheory.bi …
      -/
      ext
      /- We need to specify the type of `τ` to use the notation `⊗≫`. -/
      /-
        case h
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        u : Quiver.Hom b a
        adj : CategoryTheory.Bicategory.Adjunction f u
        x : B
        h : Quiver.Hom x a
        s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
        τ₀ : Quiver.Hom ((CategoryTheory.Bicategory.LeftLift.mk f adj.unit).whisker h) s
        ⊢ Eq τ₀.right ((fun s => CategoryTheory.Bicategory.LeftLift.homMk (CategoryThe …
      -/
      let τ : h ≫ f ⟶ s.lift := τ₀.right
      /-
        case h
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        f : Quiver.Hom a b
        u : Quiver.Hom b a
        adj : CategoryTheory.Bicategory.Adjunction f u
        x : B
        h : Quiver.Hom x a
        s : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.comp h …
        τ₀ : Quiver.Hom ((CategoryTheory.Bicategory.LeftLift.mk f adj.unit).whisker h) s
        τ : Quiver.Hom (CategoryTheory.CategoryStruct.comp h f) s.lift := τ₀.right
        ⊢ Eq τ₀.right ((fun s => CategoryTheory.Bicategory.LeftLift.homMk (CategoryThe …
      -/
      have hτ : h ◁ adj.unit ⊗≫ τ ▷ u = s.unit := by simpa [bicategoricalComp] using LeftLift.w τ₀
      calc τ
        _ = 𝟙 _ ⊗≫ h ◁ leftZigzag adj.unit adj.counit ⊗≫ τ ⊗≫ 𝟙 _ := by
          rw [adj.left_triangle]; bicategory
        _ = 𝟙 _ ⊗≫ h ◁ adj.unit ▷ f ⊗≫ (_ ◁ adj.counit ≫ τ ▷ _) ⊗≫ 𝟙 _ := by
          rw [leftZigzag]; bicategory
        _ = 𝟙 _ ⊗≫ (h ◁ adj.unit ⊗≫ τ ▷ u) ▷ f ⊗≫ s.lift ◁ adj.counit ⊗≫ 𝟙 _ := by
          rw [whisker_exchange]; bicategory
        _ = _ := by
          rw [hτ]; dsimp only [StructuredArrow.homMk_right]


/-- A left Kan lift of the identity along `u` such that `u` commutes with is a left adjoint
to `u`. The unit of this adjoint is given by the unit of the Kan lift. -/
def LeftLift.IsKan.adjunction {u : b ⟶ a} {t : LeftLift u (𝟙 a)}
    (H : IsKan t) (H' : IsKan (t.whisker u)) :
      t.lift ⊣ u :=
  let ε : u ≫ t.lift ⟶ 𝟙 b := H'.desc <| .mk _ <| (ρ_ u).hom ≫ (λ_ u).inv
  have Hε : rightZigzag t.unit ε = (ρ_ u).hom ≫ (λ_ u).inv := by
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      u : Quiver.Hom b a
      t : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.id a)
      H : t.IsKan
      H' : (t.whisker u).IsKan
      ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp u t.lift) (CategoryTheory.C …
      ⊢ Eq (CategoryTheory.Bicategory.rightZigzag t.unit ε) (CategoryTheory.Category …
    -/
    simpa [rightZigzag, bicategoricalComp] using H'.fac <| .mk _ <| (ρ_ u).hom ≫ (λ_ u).inv
    /-
      🎉 no goals
    -/
  { unit := t.unit
    counit := ε
    left_triangle := by
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        u : Quiver.Hom b a
        t : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.id a)
        H : t.IsKan
        H' : (t.whisker u).IsKan
        ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp u t.lift) (CategoryTheory.C …
        Hε : Eq (CategoryTheory.Bicategory.rightZigzag t.unit ε) (CategoryTheory.Categ …
        ⊢ Eq (CategoryTheory.Bicategory.leftZigzag t.unit ε) (CategoryTheory.CategoryS …
      -/
      apply (cancel_epi (λ_ _).inv).mp
      /-
        B : Type u
        inst✝ : CategoryTheory.Bicategory B
        a b c : B
        u : Quiver.Hom b a
        t : CategoryTheory.Bicategory.LeftLift u (CategoryTheory.CategoryStruct.id a)
        H : t.IsKan
        H' : (t.whisker u).IsKan
        ε : Quiver.Hom (CategoryTheory.CategoryStruct.comp u t.lift) (CategoryTheory.C …
        Hε : Eq (CategoryTheory.Bicategory.rightZigzag t.unit ε) (CategoryTheory.Categ …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.leftUnitor …
      -/
      apply H.hom_ext
      calc _
        _ = 𝟙 _ ⊗≫ t.unit ⊗≫ leftZigzag t.unit ε ▷ u ⊗≫ 𝟙 _ := by
          bicategory
        _ = 𝟙 _ ⊗≫ (_ ◁ t.unit ≫ t.unit ▷ _) ⊗≫ t.lift ◁ ε ▷ u ⊗≫ 𝟙 _ := by
          rw [leftZigzag]; bicategory
        _ = 𝟙 _ ⊗≫ t.unit ⊗≫ t.lift ◁ (u ◁ t.unit ⊗≫ ε ▷ u) ⊗≫ 𝟙 _ := by
          rw [whisker_exchange]; bicategory
        _ = _ := by
          rw [← rightZigzag, Hε]; bicategory
    right_triangle := Hε }


/-- For an adjuntion `f ⊣ u`, `f` is a left Kan lift of the identity along `u`.
The unit of this Kan lift is given by the unit of the adjunction. -/
def LeftLift.IsAbsKan.adjunction {u : b ⟶ a} (t : LeftLift u (𝟙 a)) (H : IsAbsKan t) :
    t.lift ⊣ u :=
  H.isKan.adjunction (H u)


theorem isRightAdjoint_TFAE (u : b ⟶ a) :
    List.TFAE [
      IsRightAdjoint u,
      HasAbsLeftKanLift u (𝟙 a),
      ∃ _ : HasLeftKanLift u (𝟙 a), LanLift.CommuteWith u (𝟙 a) u] := by
  tfae_have 1 → 2
  | h => IsAbsKan.hasAbsLeftKanLift (Adjunction.ofIsRightAdjoint u).isAbsoluteLeftKanLift
  tfae_have 2 → 3
  | h => ⟨inferInstance, inferInstance⟩
  tfae_have 3 → 1
  | ⟨h, h'⟩ => .mk <| (lanLiftIsKan u (𝟙 a)).adjunction <| LanLift.CommuteWith.isKan u (𝟙 a) u
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    u : Quiver.Hom b a
    tfae_1_to_2 : CategoryTheory.Bicategory.IsRightAdjoint u → CategoryTheory.Bica …
    tfae_2_to_3 : CategoryTheory.Bicategory.HasAbsLeftKanLift u (CategoryTheory.Ca …
    tfae_3_to_1 : (Exists fun x => CategoryTheory.Bicategory.LanLift.CommuteWith u …
    ⊢ (List.cons (CategoryTheory.Bicategory.IsRightAdjoint u) (List.cons (Category …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- A left adjoint commutes with a left Kan extension. -/
def isKanOfWhiskerLeftAdjoint
    {f : a ⟶ b} {g : a ⟶ c} {t : LeftExtension f g} (H : LeftExtension.IsKan t)
      {x : B} {h : c ⟶ x} {u : x ⟶ c} (adj : h ⊣ u) :
        LeftExtension.IsKan (t.whisker h) :=
  let η' := adj.unit
  let H' : LeftLift.IsAbsKan (.mk _ η') := adj.isAbsoluteLeftKanLift
  .mk (fun s ↦
    let k := s.extension
    let θ := s.unit
    let sτ := LeftExtension.mk _ <| 𝟙 _ ⊗≫ g ◁ η' ⊗≫ θ ▷ u ⊗≫ 𝟙 _
    let τ : t.extension ⟶ k ≫ u := H.desc sτ
    let sσ := LeftLift.mk _ <| (ρ_ _).hom ≫ τ
    let σ : t.extension ≫ h ⟶ k := H'.desc sσ
    LeftExtension.homMk σ <| (H' g).hom_ext <| by
      have Hσ : t.extension ◁ η' ⊗≫ σ ▷ u  = 𝟙 _ ⊗≫ τ := by
        simpa [bicategoricalComp] using (H' _).fac (.mk _ <| (ρ_ _).hom ≫ τ)
      dsimp only [LeftLift.whisker_lift, StructuredArrow.mk_right, LeftLift.whisker_unit,
        StructuredArrow.mk_hom_eq_self, whisker_extension, whisker_unit]
      calc _
        _ = (g ◁ η' ≫ t.unit ▷ (h ≫ u)) ⊗≫ f ◁ σ ▷ u ⊗≫ 𝟙 _ := by
          bicategory
        _ = t.unit ▷ (𝟙 c) ⊗≫ f ◁ (t.extension ◁ η' ⊗≫ σ ▷ u) ⊗≫ 𝟙 _ := by
          rw [whisker_exchange]; bicategory
        _ = (ρ_ g).hom ≫ t.unit ≫ f ◁ H.desc sτ ≫ (α_ f s.extension u).inv := by
          rw [Hσ]
          dsimp only [τ]
          bicategory
        _ = _ := by
          rw [IsKan.fac_assoc]
          dsimp only [StructuredArrow.mk_right, StructuredArrow.mk_hom_eq_self, sτ]
          bicategory) <| by
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      ⊢ ∀ (s : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStr …
          Eq τ
            ((fun s =>
                let k := s.extension;
                let θ := s.unit;
                let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTheory …
                let τ := H.desc sτ;
                let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheory.Ca …
                let σ := H'.desc sσ;
                CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
              s)
    -/
    intro s' τ₀'
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      ⊢ Eq τ₀'
          ((fun s =>
              let k := s.extension;
              let θ := s.unit;
              let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTheory.C …
              let τ := H.desc sτ;
              let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheory.Cate …
              let σ := H'.desc sσ;
              CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
            s')
    -/
    let τ' : t.extension ≫ h ⟶ s'.extension := τ₀'.right
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      ⊢ Eq τ₀'
          ((fun s =>
              let k := s.extension;
              let θ := s.unit;
              let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTheory.C …
              let τ := H.desc sτ;
              let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheory.Cate …
              let σ := H'.desc sσ;
              CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
            s')
    -/
    have Hτ' : t.unit ▷ h ⊗≫ f ◁ τ' = s'.unit := by simpa [bicategoricalComp] using τ₀'.w.symm
    /-
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq τ₀'
          ((fun s =>
              let k := s.extension;
              let θ := s.unit;
              let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTheory.C …
              let τ := H.desc sτ;
              let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheory.Cate …
              let σ := H'.desc sσ;
              CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
            s')
    -/
    ext
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq τ₀'.right
          ((fun s =>
                let k := s.extension;
                let θ := s.unit;
                let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTheory …
                let τ := H.desc sτ;
                let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheory.Ca …
                let σ := H'.desc sσ;
                CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
              s').right
    -/
    apply (H' _).hom_ext
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Bicategory.LeftLift. …
          (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Bicategory.LeftLift.m …
            (CategoryTheory.Bicategory.whiskerRight
              ((fun s =>
                    let k := s.extension;
                    let θ := s.unit;
                    let sτ := CategoryTheory.Bicategory.LeftExtension.mk (CategoryTh …
                    let τ := H.desc sτ;
                    let sσ := CategoryTheory.Bicategory.LeftLift.mk k (CategoryTheor …
                    let σ := H'.desc sσ;
                    CategoryTheory.Bicategory.LeftExtension.homMk σ ⋯)
                  s').right
              u))
    -/
    dsimp only [StructuredArrow.homMk_right]
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Bicategory.LeftLift. …
    -/
    rw [(H' _).fac]
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Bicategory.LeftLift. …
    -/
    apply (cancel_epi (ρ_ _).inv).mp
    /-
      case h
      B : Type u
      inst✝ : CategoryTheory.Bicategory B
      a b c : B
      f : Quiver.Hom a b
      g : Quiver.Hom a c
      t : CategoryTheory.Bicategory.LeftExtension f g
      H : t.IsKan
      x : B
      h : Quiver.Hom c x
      u : Quiver.Hom x c
      adj : CategoryTheory.Bicategory.Adjunction h u
      η' : Quiver.Hom (CategoryTheory.CategoryStruct.id c) (CategoryTheory.CategoryS …
      H' : (CategoryTheory.Bicategory.LeftLift.mk h η').IsAbsKan := fun {x_1} => adj …
      s' : CategoryTheory.Bicategory.LeftExtension f (CategoryTheory.CategoryStruct. …
      τ₀' : Quiver.Hom (t.whisker h) s'
      τ' : Quiver.Hom (CategoryTheory.CategoryStruct.comp t.extension h) s'.extensio …
      Hτ' : Eq (CategoryTheory.bicategoricalComp (CategoryTheory.Bicategory.whiskerR …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.rightUnito …
    -/
    apply H.hom_ext
    dsimp only [LeftLift.whisker_lift, StructuredArrow.mk_right, LeftLift.whisker_unit,
      StructuredArrow.mk_hom_eq_self]
    let σs' := LeftExtension.mk (s'.extension ≫ u)
      (𝟙 g ⊗≫ g ◁ η' ⊗≫ s'.unit ▷ u ⊗≫ 𝟙 (f ≫ s'.extension ≫ u))
    calc _
      _ = 𝟙 _ ⊗≫ (t.unit ▷ (𝟙 c) ≫ (f ≫ t.extension) ◁ η') ⊗≫ f ◁ τ' ▷ u := by
        bicategory
      _ = 𝟙 g ⊗≫ g ◁ η' ⊗≫ (t.unit ▷ h ⊗≫ f ◁ τ') ▷ u ⊗≫ 𝟙 _ := by
        rw [← whisker_exchange]; bicategory
      _ = t.unit ≫ f ◁ H.desc σs' := by
        rw [Hτ', IsKan.fac]
        dsimp only [StructuredArrow.mk_hom_eq_self, σs']
      _ = _ := by
        bicategory


instance {f : a ⟶ b} {g : a ⟶ c} {x : B} {h : c ⟶ x} [IsLeftAdjoint h] [HasLeftKanExtension f g] :
    Lan.CommuteWith f g h :=
  ⟨⟨isKanOfWhiskerLeftAdjoint (lanIsKan f g) (Adjunction.ofIsLeftAdjoint h)⟩⟩


