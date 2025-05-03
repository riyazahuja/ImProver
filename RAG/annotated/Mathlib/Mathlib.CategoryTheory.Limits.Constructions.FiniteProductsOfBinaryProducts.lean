/--
Given `n+1` objects of `C`, a fan for the last `n` with point `c₁.pt` and
a binary fan on `c₁.pt` and `f 0`, we can build a fan for all `n+1`.

In `extendFanIsLimit` we show that if the two given fans are limits, then this fan is also a
limit.
-/
@[simps!] -- Porting note: removed semi-reducible config
def extendFan {n : ℕ} {f : Fin (n + 1) → C} (c₁ : Fan fun i : Fin n => f i.succ)
    (c₂ : BinaryFan (f 0) c₁.pt) : Fan f :=
  Fan.mk c₂.pt
    (by
      /-
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        ⊢ (b : Fin (HAdd.hAdd n 1)) → Quiver.Hom c₂.pt (f b)
      -/
      refine Fin.cases ?_ ?_
        /-
          case refine_1
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
          ⊢ Quiver.Hom c₂.pt (f 0)
        -/
      · apply c₂.fst
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
          ⊢ (i : Fin n) → Quiver.Hom c₂.pt (f i.succ)
        -/
      · intro i
        /-
          case refine_2
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
          i : Fin n
          ⊢ Quiver.Hom c₂.pt (f i.succ)
        -/
        apply c₂.snd ≫ c₁.π.app ⟨i⟩)
        /-
          🎉 no goals
        -/


/-- Show that if the two given fans in `extendFan` are limits, then the constructed fan is also a
limit.
-/
def extendFanIsLimit {n : ℕ} (f : Fin (n + 1) → C) {c₁ : Fan fun i : Fin n => f i.succ}
    {c₂ : BinaryFan (f 0) c₁.pt} (t₁ : IsLimit c₁) (t₂ : IsLimit c₂) :
    IsLimit (extendFan c₁ c₂) where
  lift s := by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsLimit c₁
      t₂ : CategoryTheory.Limits.IsLimit c₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      ⊢ Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
    -/
    apply (BinaryFan.IsLimit.lift' t₂ (s.π.app ⟨0⟩) _).1
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsLimit c₁
      t₂ : CategoryTheory.Limits.IsLimit c₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      ⊢ Quiver.Hom (((CategoryTheory.Functor.const (CategoryTheory.Discrete (Fin (HA …
    -/
    apply t₁.lift ⟨_, Discrete.natTrans fun ⟨i⟩ => s.π.app ⟨i.succ⟩⟩
    /-
      🎉 no goals
    -/
  fac := fun s ⟨j⟩ => by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsLimit c₁
      t₂ : CategoryTheory.Limits.IsLimit c₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => ↑(CategoryTheory.Limits.Bi …
    -/
    refine Fin.inductionOn j ?_ ?_
      /-
        case refine_1
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => ↑(CategoryTheory.Limits.Bi …
      -/
    · apply (BinaryFan.IsLimit.lift' t₂ _ _).2.1
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
        j : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i : Fin n), Eq (CategoryTheory.CategoryStruct.comp ((fun s => ↑(CategoryT …
      -/
    · rintro i -
      /-
        case refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => ↑(CategoryTheory.Limits.Bi …
      -/
      dsimp only [extendFan_π_app]
      /-
        case refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑(CategoryTheory.Limits.BinaryFan.Is …
      -/
      rw [Fin.cases_succ, ← assoc, (BinaryFan.IsLimit.lift' t₂ _ _).2.2, t₁.fac]
      /-
        case refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        x✝ : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq ({ pt := s.pt, π := CategoryTheory.Discrete.natTrans fun x => s.π.app { a …
      -/
      rfl
      /-
        🎉 no goals
      -/
  uniq s m w := by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsLimit c₁
      t₂ : CategoryTheory.Limits.IsLimit c₂
      s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
      w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
      ⊢ Eq m ((fun s => ↑(CategoryTheory.Limits.BinaryFan.IsLimit.lift' t₂ (s.π.app  …
    -/
    apply BinaryFan.IsLimit.hom_ext t₂
      /-
        case h₁
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m c₂.fst) (CategoryTheory.CategoryStr …
      -/
    · rw [(BinaryFan.IsLimit.lift' t₂ _ _).2.1]
      /-
        case h₁
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m c₂.fst) (s.π.app { as := 0 })
      -/
      apply w ⟨0⟩
      /-
        🎉 no goals
      -/
      /-
        case h₂
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m c₂.snd) (CategoryTheory.CategoryStr …
      -/
    · rw [(BinaryFan.IsLimit.lift' t₂ _ _).2.2]
      /-
        case h₂
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m c₂.snd) (t₁.lift { pt := s.pt, π := …
      -/
      apply t₁.uniq ⟨_, _⟩
      /-
        case h₂.x
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ ∀ (j : CategoryTheory.Discrete (Fin n)), Eq (CategoryTheory.CategoryStruct.c …
      -/
      rintro ⟨j⟩
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
      -/
      rw [assoc]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      dsimp only [Discrete.natTrans_app]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      rw [← w ⟨j.succ⟩]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      dsimp only [extendFan_π_app]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Fan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryFan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom s.pt (CategoryTheory.extendFan c₁ c₂).pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
      -/
      rw [Fin.cases_succ]
      /-
        🎉 no goals
      -/


/-- If `C` has a terminal object and binary products, then it has a product for objects indexed by
`Fin n`.
This is a helper lemma for `hasFiniteProductsOfHasBinaryAndTerminal`, which is more general
than this.
-/
private theorem hasProduct_fin : ∀ (n : ℕ) (f : Fin n → C), HasProduct f
  | 0 => fun _ =>
    letI : HasLimitsOfShape (Discrete (Fin 0)) C :=
      hasLimitsOfShape_of_equivalence (Discrete.equivalence.{0} finZeroEquiv'.symm)
    inferInstance
  | n + 1 => fun f =>
    haveI := hasProduct_fin n
    HasLimit.mk ⟨_, extendFanIsLimit f (limit.isLimit _) (limit.isLimit _)⟩


/-- If `C` has a terminal object and binary products, then it has finite products. -/
theorem hasFiniteProducts_of_has_binary_and_terminal : HasFiniteProducts C :=
  ⟨fun n => ⟨fun K =>
    let this := hasProduct_fin n fun n => K.obj ⟨n⟩
    let that : (Discrete.functor fun n => K.obj ⟨n⟩) ≅ K := Discrete.natIso fun ⟨_⟩ => Iso.refl _
    @hasLimitOfIso _ _ _ _ _ _ this that⟩⟩



/-- If `F` preserves the terminal object and binary products, then it preserves products indexed by
`Fin n` for any `n`.
-/
lemma preservesFinOfPreservesBinaryAndTerminal :
    ∀ (n : ℕ) (f : Fin n → C), PreservesLimit (Discrete.functor f) F
  | 0 => fun f => by
    letI : PreservesLimitsOfShape (Discrete (Fin 0)) F :=
      preservesLimitsOfShape_of_equiv.{0, 0} (Discrete.equivalence finZeroEquiv'.symm) _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      f : Fin 0 → C
      this : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete ( …
      ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor f) F
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      ⊢ ∀ (f : Fin (HAdd.hAdd n 1) → C), CategoryTheory.Limits.PreservesLimit (Categ …
    -/
    haveI := preservesFinOfPreservesBinaryAndTerminal n
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
      ⊢ ∀ (f : Fin (HAdd.hAdd n 1) → C), CategoryTheory.Limits.PreservesLimit (Categ …
    -/
    intro f
    apply
      preservesLimit_of_preserves_limit_cone
        (extendFanIsLimit f (limit.isLimit _) (limit.isLimit _)) _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheory …
      f : Fin (HAdd.hAdd n 1) → C
      ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.extendFan (Category …
    -/
    apply (isLimitMapConeFanMkEquiv _ _ _).symm _
    let this :=
      extendFanIsLimit (fun i => F.obj (f i)) (isLimitOfHasProductOfPreservesLimit F _)
        (isLimitOfHasBinaryProductOfPreservesLimit F _ _)
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fan.mk (F.obj (Category …
    -/
    refine IsLimit.ofIsoLimit this ?_
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
      ⊢ CategoryTheory.Iso (CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk ( …
    -/
    apply Cones.ext _ _
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        ⊢ CategoryTheory.Iso (CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk ( …
      -/
    · apply Iso.refl _
      /-
        🎉 no goals
      -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
      ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq ((CategoryTheory.e …
    -/
    rintro ⟨j⟩
    /-
      case mk
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Category …
    -/
    refine Fin.inductionOn j ?_ ?_
      /-
        case mk.refine_1
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Category …
      -/
    · apply (Category.id_comp _).symm
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i : Fin n), Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk ( …
      -/
    · rintro i _
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Categ …
        ⊢ Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Category …
      -/
      dsimp [extendFan_π_app, Iso.refl_hom, Fan.mk_π_app]
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Categ …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.prod.snd …
      -/
      change F.map _ ≫ _ = 𝟙 _ ≫ _
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Categ …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.prod.snd …
      -/
      simp only [id_comp, ← F.map_comp]
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.HasFiniteProducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesLimit (CategoryTheor …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsLimit (CategoryTheory.extendFan (CategoryTheory …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq ((CategoryTheory.extendFan (CategoryTheory.Limits.Fan.mk (F.obj (Categ …
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.snd …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- If `F` preserves the terminal object and binary products, then it preserves limits of shape
`Discrete (Fin n)`.
-/
lemma preservesShape_fin_of_preserves_binary_and_terminal (n : ℕ) :
    PreservesLimitsOfShape (Discrete (Fin n)) F where
  preservesLimit {K} := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    let that : (Discrete.functor fun n => K.obj ⟨n⟩) ≅ K := Discrete.natIso fun ⟨i⟩ => Iso.refl _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      that : CategoryTheory.Iso (CategoryTheory.Discrete.functor fun n_1 => K.obj {  …
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    haveI := preservesFinOfPreservesBinaryAndTerminal F n fun n => K.obj ⟨n⟩
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      that : CategoryTheory.Iso (CategoryTheory.Discrete.functor fun n_1 => K.obj {  …
      this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor f …
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    apply preservesLimit_of_iso_diagram F that
    /-
      🎉 no goals
    -/


/-- If `F` preserves the terminal object and binary products then it preserves finite products. -/
lemma preservesFiniteProducts_of_preserves_binary_and_terminal (J : Type*) [Fintype J] :
    PreservesLimitsOfShape (Discrete J) F := by
  classical
    let e := Fintype.equivFin J
    haveI := preservesShape_fin_of_preserves_binary_and_terminal F (Fintype.card J)
    apply preservesLimitsOfShape_of_equiv (Discrete.equivalence e).symm


/-- Given `n+1` objects of `C`, a cofan for the last `n` with point `c₁.pt`
and a binary cofan on `c₁.X` and `f 0`, we can build a cofan for all `n+1`.

In `extendCofanIsColimit` we show that if the two given cofans are colimits,
then this cofan is also a colimit.
-/

@[simps!] -- Porting note: removed semireducible config
def extendCofan {n : ℕ} {f : Fin (n + 1) → C} (c₁ : Cofan fun i : Fin n => f i.succ)
    (c₂ : BinaryCofan (f 0) c₁.pt) : Cofan f :=
  Cofan.mk c₂.pt
    (by
      /-
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        ⊢ (b : Fin (HAdd.hAdd n 1)) → Quiver.Hom (f b) c₂.pt
      -/
      refine Fin.cases ?_ ?_
        /-
          case refine_1
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
          ⊢ Quiver.Hom (f 0) c₂.pt
        -/
      · apply c₂.inl
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
          ⊢ (i : Fin n) → Quiver.Hom (f i.succ) c₂.pt
        -/
      · intro i
        /-
          case refine_2
          J : Type v
          inst✝² : CategoryTheory.SmallCategory J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          n : Nat
          f : Fin (HAdd.hAdd n 1) → C
          c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
          c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
          i : Fin n
          ⊢ Quiver.Hom (f i.succ) c₂.pt
        -/
        apply c₁.ι.app ⟨i⟩ ≫ c₂.inr)
        /-
          🎉 no goals
        -/


/-- Show that if the two given cofans in `extendCofan` are colimits,
then the constructed cofan is also a colimit.
-/
def extendCofanIsColimit {n : ℕ} (f : Fin (n + 1) → C) {c₁ : Cofan fun i : Fin n => f i.succ}
    {c₂ : BinaryCofan (f 0) c₁.pt} (t₁ : IsColimit c₁) (t₂ : IsColimit c₂) :
    IsColimit (extendCofan c₁ c₂) where
  desc s := by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      ⊢ Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
    -/
    apply (BinaryCofan.IsColimit.desc' t₂ (s.ι.app ⟨0⟩) _).1
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      ⊢ Quiver.Hom c₁.pt (((CategoryTheory.Functor.const (CategoryTheory.Discrete (F …
    -/
    apply t₁.desc ⟨_, Discrete.natTrans fun i => s.ι.app ⟨i.as.succ⟩⟩
    /-
      🎉 no goals
    -/
  fac s := by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory.Ca …
    -/
    rintro ⟨j⟩
    /-
      case mk
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan c₁ c₂).ι …
    -/
    refine Fin.inductionOn j ?_ ?_
      /-
        case mk.refine_1
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan c₁ c₂).ι …
      -/
    · apply (BinaryCofan.IsColimit.desc' t₂ _ _).2.1
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        j : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i : Fin n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.exten …
      -/
    · rintro i -
      /-
        case mk.refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan c₁ c₂).ι …
      -/
      dsimp only [extendCofan_ι_app]
      /-
        case mk.refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Fin.cases c₂.inl (fun i => CategoryT …
      -/
      rw [Fin.cases_succ, assoc, (BinaryCofan.IsColimit.desc' t₂ _ _).2.2, t₁.fac]
      /-
        case mk.refine_2
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Eq ({ pt := s.pt, ι := CategoryTheory.Discrete.natTrans fun i => s.ι.app { a …
      -/
      rfl
      /-
        🎉 no goals
      -/
  uniq s m w := by
    /-
      J : Type v
      inst✝² : CategoryTheory.SmallCategory J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      n : Nat
      f : Fin (HAdd.hAdd n 1) → C
      c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
      c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
      w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
      ⊢ Eq m ((fun s => ↑(CategoryTheory.Limits.BinaryCofan.IsColimit.desc' t₂ (s.ι. …
    -/
    apply BinaryCofan.IsColimit.hom_ext t₂
      /-
        case h₁
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c₂.inl m) (CategoryTheory.CategoryStr …
      -/
    · rw [(BinaryCofan.IsColimit.desc' t₂ _ _).2.1]
      /-
        case h₁
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c₂.inl m) (s.ι.app { as := 0 })
      -/
      apply w ⟨0⟩
      /-
        🎉 no goals
      -/
      /-
        case h₂
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c₂.inr m) (CategoryTheory.CategoryStr …
      -/
    · rw [(BinaryCofan.IsColimit.desc' t₂ _ _).2.2]
      /-
        case h₂
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c₂.inr m) (t₁.desc { pt := s.pt, ι := …
      -/
      apply t₁.uniq ⟨_, _⟩
      /-
        case h₂.x
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        ⊢ ∀ (j : CategoryTheory.Discrete (Fin n)), Eq (CategoryTheory.CategoryStruct.c …
      -/
      rintro ⟨j⟩
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app { as := j }) (CategoryTheor …
      -/
      dsimp only [Discrete.natTrans_app]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app { as := j }) (CategoryTheor …
      -/
      rw [← w ⟨j.succ⟩]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app { as := j }) (CategoryTheor …
      -/
      dsimp only [extendCofan_ι_app]
      /-
        case h₂.x.mk
        J : Type v
        inst✝² : CategoryTheory.SmallCategory J
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝ : CategoryTheory.Category.{v', u'} D
        n : Nat
        f : Fin (HAdd.hAdd n 1) → C
        c₁ : CategoryTheory.Limits.Cofan fun i => f i.succ
        c₂ : CategoryTheory.Limits.BinaryCofan (f 0) c₁.pt
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
        m : Quiver.Hom (CategoryTheory.extendCofan c₁ c₂).pt s.pt
        w : ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory. …
        j : Fin n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app { as := j }) (CategoryTheor …
      -/
      rw [Fin.cases_succ, assoc]
      /-
        🎉 no goals
      -/


/--
If `C` has an initial object and binary coproducts, then it has a coproduct for objects indexed by
`Fin n`.
This is a helper lemma for `hasCofiniteProductsOfHasBinaryAndTerminal`, which is more general
than this.
-/
private theorem hasCoproduct_fin : ∀ (n : ℕ) (f : Fin n → C), HasCoproduct f
  | 0 => fun _ =>
    letI : HasColimitsOfShape (Discrete (Fin 0)) C :=
      hasColimitsOfShape_of_equivalence (Discrete.equivalence.{0} finZeroEquiv'.symm)
    inferInstance
  | n + 1 => fun f =>
    haveI := hasCoproduct_fin n
    HasColimit.mk ⟨_, extendCofanIsColimit f (colimit.isColimit _) (colimit.isColimit _)⟩


/-- If `C` has an initial object and binary coproducts, then it has finite coproducts. -/
theorem hasFiniteCoproducts_of_has_binary_and_initial : HasFiniteCoproducts C :=
  ⟨fun n => ⟨fun K =>
    letI := hasCoproduct_fin n fun n => K.obj ⟨n⟩
    let that : K ≅ Discrete.functor fun n => K.obj ⟨n⟩ := Discrete.natIso fun ⟨_⟩ => Iso.refl _
    @hasColimitOfIso _ _ _ _ _ _ this that⟩⟩


/-- If `F` preserves the initial object and binary coproducts, then it preserves products indexed by
`Fin n` for any `n`.
-/
lemma preserves_fin_of_preserves_binary_and_initial :
    ∀ (n : ℕ) (f : Fin n → C), PreservesColimit (Discrete.functor f) F
  | 0 => fun f => by
    letI : PreservesColimitsOfShape (Discrete (Fin 0)) F :=
      preservesColimitsOfShape_of_equiv.{0, 0} (Discrete.equivalence finZeroEquiv'.symm) _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      f : Fin 0 → C
      this : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete …
      ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functor f) F
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      ⊢ ∀ (f : Fin (HAdd.hAdd n 1) → C), CategoryTheory.Limits.PreservesColimit (Cat …
    -/
    haveI := preserves_fin_of_preserves_binary_and_initial n
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryTheo …
      ⊢ ∀ (f : Fin (HAdd.hAdd n 1) → C), CategoryTheory.Limits.PreservesColimit (Cat …
    -/
    intro f
    apply
      preservesColimit_of_preserves_colimit_cocone
        (extendCofanIsColimit f (colimit.isColimit _) (colimit.isColimit _)) _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryTheo …
      f : Fin (HAdd.hAdd n 1) → C
      ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.extendCofan (Ca …
    -/
    apply (isColimitMapCoconeCofanMkEquiv _ _ _).symm _
    let this :=
      extendCofanIsColimit (fun i => F.obj (f i))
        (isColimitOfHasCoproductOfPreservesColimit F _)
        (isColimitOfHasBinaryCoproductOfPreservesColimit F _ _)
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (F.obj (Cate …
    -/
    refine IsColimit.ofIsoColimit this ?_
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
      ⊢ CategoryTheory.Iso (CategoryTheory.extendCofan (CategoryTheory.Limits.Cofan. …
    -/
    apply Cocones.ext _ _
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
        ⊢ CategoryTheory.Iso (CategoryTheory.extendCofan (CategoryTheory.Limits.Cofan. …
      -/
    · apply Iso.refl _
      /-
        🎉 no goals
      -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
      ⊢ ∀ (j : CategoryTheory.Discrete (Fin (HAdd.hAdd n 1))), Eq (CategoryTheory.Ca …
    -/
    rintro ⟨j⟩
    /-
      case mk
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
      f : Fin (HAdd.hAdd n 1) → C
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan (Categor …
    -/
    refine Fin.inductionOn j ?_ ?_
      /-
        case mk.refine_1
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan (Categor …
      -/
    · apply Category.comp_id
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
        j : Fin (HAdd.hAdd n 1)
        ⊢ ∀ (i : Fin n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.exten …
      -/
    · rintro i _
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan (Cate …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan (Categor …
      -/
      dsimp [extendCofan_ι_app, Iso.refl_hom, Cofan.mk_ι_app]
      /-
        case mk.refine_2
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        n : Nat
        this✝ : ∀ (f : Fin n → C), CategoryTheory.Limits.PreservesColimit (CategoryThe …
        f : Fin (HAdd.hAdd n 1) → C
        this : CategoryTheory.Limits.IsColimit (CategoryTheory.extendCofan (CategoryTh …
        j : Fin (HAdd.hAdd n 1)
        i : Fin n
        a✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.extendCofan (Cate …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      rw [comp_id, ← F.map_comp]
      /-
        🎉 no goals
      -/


/-- If `F` preserves the initial object and binary coproducts, then it preserves colimits of shape
`Discrete (Fin n)`.
-/
lemma preservesShape_fin_of_preserves_binary_and_initial (n : ℕ) :
    PreservesColimitsOfShape (Discrete (Fin n)) F where
  preservesColimit {K} := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      ⊢ CategoryTheory.Limits.PreservesColimit K F
    -/
    let that : (Discrete.functor fun n => K.obj ⟨n⟩) ≅ K := Discrete.natIso fun ⟨i⟩ => Iso.refl _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      that : CategoryTheory.Iso (CategoryTheory.Discrete.functor fun n_1 => K.obj {  …
      ⊢ CategoryTheory.Limits.PreservesColimit K F
    -/
    haveI := preserves_fin_of_preserves_binary_and_initial F n fun n => K.obj ⟨n⟩
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      n : Nat
      K : CategoryTheory.Functor (CategoryTheory.Discrete (Fin n)) C
      that : CategoryTheory.Iso (CategoryTheory.Discrete.functor fun n_1 => K.obj {  …
      this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functor …
      ⊢ CategoryTheory.Limits.PreservesColimit K F
    -/
    apply preservesColimit_of_iso_diagram F that
    /-
      🎉 no goals
    -/


/-- If `F` preserves the initial object and binary coproducts then it preserves finite products. -/
lemma preservesFiniteCoproductsOfPreservesBinaryAndInitial (J : Type*) [Fintype J] :
    PreservesColimitsOfShape (Discrete J) F := by
  classical
    let e := Fintype.equivFin J
    haveI := preservesShape_fin_of_preserves_binary_and_initial F (Fintype.card J)
    apply preservesColimitsOfShape_of_equiv (Discrete.equivalence e).symm


