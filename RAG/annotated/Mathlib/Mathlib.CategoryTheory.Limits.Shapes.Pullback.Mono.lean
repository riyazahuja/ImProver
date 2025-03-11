/-- Monomorphisms are stable under pullback in the first argument. -/
theorem mono_snd_of_is_pullback_of_mono {t : PullbackCone f g} (ht : IsLimit t) [Mono f] :
    Mono t.snd := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.Mono t.snd
  -/
  refine ⟨fun {W} h k i => IsLimit.hom_ext ht ?_ i⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono f
    W : C
    h k : Quiver.Hom W t.pt
    i : Eq (CategoryTheory.CategoryStruct.comp h t.snd) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h t.fst) (CategoryTheory.CategoryStru …
  -/
  rw [← cancel_mono f, Category.assoc, Category.assoc, condition]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono f
    W : C
    h k : Quiver.Hom W t.pt
    i : Eq (CategoryTheory.CategoryStruct.comp h t.snd) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
  -/
  apply reassoc_of% i
  /-
    🎉 no goals
  -/


/-- Monomorphisms are stable under pullback in the second argument. -/
theorem mono_fst_of_is_pullback_of_mono {t : PullbackCone f g} (ht : IsLimit t) [Mono g] :
    Mono t.fst := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono g
    ⊢ CategoryTheory.Mono t.fst
  -/
  refine ⟨fun {W} h k i => IsLimit.hom_ext ht i ?_⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono g
    W : C
    h k : Quiver.Hom W t.pt
    i : Eq (CategoryTheory.CategoryStruct.comp h t.fst) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h t.snd) (CategoryTheory.CategoryStru …
  -/
  rw [← cancel_mono g, Category.assoc, Category.assoc, ← condition]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    inst✝ : CategoryTheory.Mono g
    W : C
    h k : Quiver.Hom W t.pt
    i : Eq (CategoryTheory.CategoryStruct.comp h t.fst) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
  -/
  apply reassoc_of% i
  /-
    🎉 no goals
  -/


/--
The pullback cone `(𝟙 X, 𝟙 X)` for the pair `(f, f)` is a limit if `f` is a mono. The converse is
shown in `mono_of_pullback_is_id`.
-/
def isLimitMkIdId (f : X ⟶ Y) [Mono f] : IsLimit (mk (𝟙 X) (𝟙 X) rfl : PullbackCone f f) :=
  IsLimit.mk _ (fun s => s.fst) (fun _ => Category.comp_id _)
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   W X Y Z : C
                   f✝ : Quiver.Hom X Z
                   g : Quiver.Hom Y Z
                   f : Quiver.Hom X Y
                   inst✝ : CategoryTheory.Mono f
                   s : CategoryTheory.Limits.PullbackCone f f
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => s.fst) s) (CategoryTheory. …
                 -/
    (fun s => by rw [← cancel_mono f, Category.comp_id, s.condition]) fun s m m₁ _ => by
                 /-
                   🎉 no goals
                 -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      s : CategoryTheory.Limits.PullbackCone f f
      m : Quiver.Hom s.pt X
      m₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.i …
      x✝ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.i …
      ⊢ Eq m ((fun s => s.fst) s)
    -/
    simpa using m₁
    /-
      🎉 no goals
    -/


/--
`f` is a mono if the pullback cone `(𝟙 X, 𝟙 X)` is a limit for the pair `(f, f)`. The converse is
given in `PullbackCone.is_id_of_mono`.
-/
theorem mono_of_isLimitMkIdId (f : X ⟶ Y) (t : IsLimit (mk (𝟙 X) (𝟙 X) rfl : PullbackCone f f)) :
    Mono f :=
  ⟨fun {Z} g h eq => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Cate …
      Z : C
      g h : Quiver.Hom Z X
      eq : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    rcases PullbackCone.IsLimit.lift' t _ _ eq with ⟨_, rfl, rfl⟩
    /-
      case mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Cate …
      Z : C
      val✝ : Quiver.Hom Z (CategoryTheory.Limits.PullbackCone.mk (CategoryTheory.Cat …
      eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp val✝ (CategoryTheory.Limits.PullbackC …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


/-- Suppose `f` and `g` are two morphisms with a common codomain and `s` is a limit cone over the
    diagram formed by `f` and `g`. Suppose `f` and `g` both factor through a monomorphism `h` via
    `x` and `y`, respectively.  Then `s` is also a limit cone over the diagram formed by `x` and
    `y`. -/
def isLimitOfFactors (f : X ⟶ Z) (g : Y ⟶ Z) (h : W ⟶ Z) [Mono h] (x : X ⟶ W) (y : Y ⟶ W)
    (hxh : x ≫ h = f) (hyh : y ≫ h = g) (s : PullbackCone f g) (hs : IsLimit s) :
    IsLimit
      (PullbackCone.mk _ _
        (show s.fst ≫ x = s.snd ≫ y from
                                  /-
                                    C : Type u
                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                    W X Y Z : C
                                    f✝ : Quiver.Hom X Z
                                    g✝ : Quiver.Hom Y Z
                                    f : Quiver.Hom X Z
                                    g : Quiver.Hom Y Z
                                    h : Quiver.Hom W Z
                                    inst✝ : CategoryTheory.Mono h
                                    x : Quiver.Hom X W
                                    y : Quiver.Hom Y W
                                    hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
                                    hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
                                    s : CategoryTheory.Limits.PullbackCone f g
                                    hs : CategoryTheory.Limits.IsLimit s
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
                                  -/
          (cancel_mono h).1 <| by simp only [Category.assoc, hxh, hyh, s.condition])) :=
                                  /-
                                    🎉 no goals
                                  -/
  PullbackCone.isLimitAux' _ fun t =>
    have : fst t ≫ x ≫ h = snd t ≫ y ≫ h := by  -- Porting note: reassoc workaround
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f✝ : Quiver.Hom X Z
        g✝ : Quiver.Hom Y Z
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        h : Quiver.Hom W Z
        inst✝ : CategoryTheory.Mono h
        x : Quiver.Hom X W
        y : Quiver.Hom Y W
        hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
        hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
        s : CategoryTheory.Limits.PullbackCone f g
        hs : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.PullbackCone x y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategoryStruct. …
      -/
      rw [← Category.assoc, ← Category.assoc]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f✝ : Quiver.Hom X Z
        g✝ : Quiver.Hom Y Z
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        h : Quiver.Hom W Z
        inst✝ : CategoryTheory.Mono h
        x : Quiver.Hom X W
        y : Quiver.Hom Y W
        hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
        hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
        s : CategoryTheory.Limits.PullbackCone f g
        hs : CategoryTheory.Limits.IsLimit s
        t : CategoryTheory.Limits.PullbackCone x y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
      -/
      apply congrArg (· ≫ h) t.condition
      /-
        🎉 no goals
      -/
                                                /-
                                                  C : Type u
                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                  W X Y Z : C
                                                  f✝ : Quiver.Hom X Z
                                                  g✝ : Quiver.Hom Y Z
                                                  f : Quiver.Hom X Z
                                                  g : Quiver.Hom Y Z
                                                  h : Quiver.Hom W Z
                                                  inst✝ : CategoryTheory.Mono h
                                                  x : Quiver.Hom X W
                                                  y : Quiver.Hom Y W
                                                  hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
                                                  hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
                                                  s : CategoryTheory.Limits.PullbackCone f g
                                                  hs : CategoryTheory.Limits.IsLimit s
                                                  t : CategoryTheory.Limits.PullbackCone x y
                                                  this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp t.fst f) (CategoryTheory.CategoryStru …
                                                -/
    ⟨hs.lift (PullbackCone.mk t.fst t.snd <| by rw [← hxh, ← hyh, this]),
                                                /-
                                                  🎉 no goals
                                                -/
      ⟨hs.fac _ WalkingCospan.left, hs.fac _ WalkingCospan.right, fun hr hr' => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f✝ : Quiver.Hom X Z
          g✝ : Quiver.Hom Y Z
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          h : Quiver.Hom W Z
          inst✝ : CategoryTheory.Mono h
          x : Quiver.Hom X W
          y : Quiver.Hom Y W
          hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
          hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
          s : CategoryTheory.Limits.PullbackCone f g
          hs : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.PullbackCone x y
          this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
          m✝ : Quiver.Hom t.pt (CategoryTheory.Limits.PullbackCone.mk s.fst s.snd ⋯).pt
          hr : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
          hr' : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullbac …
          ⊢ Eq m✝ (hs.lift (CategoryTheory.Limits.PullbackCone.mk t.fst t.snd ⋯))
        -/
        apply PullbackCone.IsLimit.hom_ext hs <;>
              /-
                case h₀
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                W X Y Z : C
                f✝ : Quiver.Hom X Z
                g✝ : Quiver.Hom Y Z
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                h : Quiver.Hom W Z
                inst✝ : CategoryTheory.Mono h
                x : Quiver.Hom X W
                y : Quiver.Hom Y W
                hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
                hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
                s : CategoryTheory.Limits.PullbackCone f g
                hs : CategoryTheory.Limits.IsLimit s
                t : CategoryTheory.Limits.PullbackCone x y
                this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
                m✝ : Quiver.Hom t.pt (CategoryTheory.Limits.PullbackCone.mk s.fst s.snd ⋯).pt
                hr : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullback …
                hr' : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Pullbac …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ s.fst) (CategoryTheory.CategoryStr …
              -/
              simp only [PullbackCone.mk_fst, PullbackCone.mk_snd] at hr hr' ⊢ <;>
            /-
              case h₀
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              W X Y Z : C
              f✝ : Quiver.Hom X Z
              g✝ : Quiver.Hom Y Z
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              h : Quiver.Hom W Z
              inst✝ : CategoryTheory.Mono h
              x : Quiver.Hom X W
              y : Quiver.Hom Y W
              hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
              hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
              s : CategoryTheory.Limits.PullbackCone f g
              hs : CategoryTheory.Limits.IsLimit s
              t : CategoryTheory.Limits.PullbackCone x y
              this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
              m✝ : Quiver.Hom t.pt (CategoryTheory.Limits.PullbackCone.mk s.fst s.snd ⋯).pt
              hr : Eq (CategoryTheory.CategoryStruct.comp m✝ s.fst) t.fst
              hr' : Eq (CategoryTheory.CategoryStruct.comp m✝ s.snd) t.snd
              ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ s.fst) (CategoryTheory.CategoryStr …
            -/
            simp only [hr, hr'] <;>
          /-
            case h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Z
            g✝ : Quiver.Hom Y Z
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            h : Quiver.Hom W Z
            inst✝ : CategoryTheory.Mono h
            x : Quiver.Hom X W
            y : Quiver.Hom Y W
            hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
            hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
            s : CategoryTheory.Limits.PullbackCone f g
            hs : CategoryTheory.Limits.IsLimit s
            t : CategoryTheory.Limits.PullbackCone x y
            this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
            m✝ : Quiver.Hom t.pt (CategoryTheory.Limits.PullbackCone.mk s.fst s.snd ⋯).pt
            hr : Eq (CategoryTheory.CategoryStruct.comp m✝ s.fst) t.fst
            hr' : Eq (CategoryTheory.CategoryStruct.comp m✝ s.snd) t.snd
            ⊢ Eq t.fst (CategoryTheory.CategoryStruct.comp (hs.lift (CategoryTheory.Limits …
          -/
          symm
        /-
          case h₀
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f✝ : Quiver.Hom X Z
          g✝ : Quiver.Hom Y Z
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          h : Quiver.Hom W Z
          inst✝ : CategoryTheory.Mono h
          x : Quiver.Hom X W
          y : Quiver.Hom Y W
          hxh : Eq (CategoryTheory.CategoryStruct.comp x h) f
          hyh : Eq (CategoryTheory.CategoryStruct.comp y h) g
          s : CategoryTheory.Limits.PullbackCone f g
          hs : CategoryTheory.Limits.IsLimit s
          t : CategoryTheory.Limits.PullbackCone x y
          this : Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.CategorySt …
          m✝ : Quiver.Hom t.pt (CategoryTheory.Limits.PullbackCone.mk s.fst s.snd ⋯).pt
          hr : Eq (CategoryTheory.CategoryStruct.comp m✝ s.fst) t.fst
          hr' : Eq (CategoryTheory.CategoryStruct.comp m✝ s.snd) t.snd
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (hs.lift (CategoryTheory.Limits.Pullb …
        -/
        exacts [hs.fac _ WalkingCospan.left, hs.fac _ WalkingCospan.right]⟩⟩
        /-
          🎉 no goals
        -/


/-- If `W` is the pullback of `f, g`, it is also the pullback of `f ≫ i, g ≫ i` for any mono `i`. -/
def isLimitOfCompMono (f : X ⟶ W) (g : Y ⟶ W) (i : W ⟶ Z) [Mono i] (s : PullbackCone f g)
    (H : IsLimit s) :
    IsLimit
      (PullbackCone.mk _ _
        (show s.fst ≫ f ≫ i = s.snd ≫ g ≫ i by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Z
            g✝ : Quiver.Hom Y Z
            f : Quiver.Hom X W
            g : Quiver.Hom Y W
            i : Quiver.Hom W Z
            inst✝ : CategoryTheory.Mono i
            s : CategoryTheory.Limits.PullbackCone f g
            H : CategoryTheory.Limits.IsLimit s
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst (CategoryTheory.CategoryStruct. …
          -/
          rw [← Category.assoc, ← Category.assoc, s.condition])) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    i : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono i
    s : CategoryTheory.Limits.PullbackCone f g
    H : CategoryTheory.Limits.IsLimit s
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk s.fst s …
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    i : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono i
    s : CategoryTheory.Limits.PullbackCone f g
    H : CategoryTheory.Limits.IsLimit s
    ⊢ (s_1 : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.com …
  -/
  intro s
  rcases PullbackCone.IsLimit.lift' H s.fst s.snd
      ((cancel_mono i).mp (by simpa using s.condition)) with
    ⟨l, h₁, h₂⟩
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    i : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono i
    s✝ : CategoryTheory.Limits.PullbackCone f g
    H : CategoryTheory.Limits.IsLimit s✝
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f i …
    l : Quiver.Hom s.pt s✝.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp l s✝.fst) s.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp l s✝.snd) s.snd
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
  -/
  refine ⟨l, h₁, h₂, ?_⟩
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    i : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono i
    s✝ : CategoryTheory.Limits.PullbackCone f g
    H : CategoryTheory.Limits.IsLimit s✝
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f i …
    l : Quiver.Hom s.pt s✝.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp l s✝.fst) s.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp l s✝.snd) s.snd
    ⊢ ∀ {m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk s✝.fst s✝.snd  …
  -/
  intro m hm₁ hm₂
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    f : Quiver.Hom X W
    g : Quiver.Hom Y W
    i : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono i
    s✝ : CategoryTheory.Limits.PullbackCone f g
    H : CategoryTheory.Limits.IsLimit s✝
    s : CategoryTheory.Limits.PullbackCone (CategoryTheory.CategoryStruct.comp f i …
    l : Quiver.Hom s.pt s✝.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp l s✝.fst) s.fst
    h₂ : Eq (CategoryTheory.CategoryStruct.comp l s✝.snd) s.snd
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk s✝.fst s✝.snd ⋯).pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Pullback …
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Pullback …
    ⊢ Eq m l
  -/
  exact (PullbackCone.IsLimit.hom_ext H (hm₁.trans h₁.symm) (hm₂.trans h₂.symm) : _)
  /-
    🎉 no goals
  -/


/-- The pullback of a monomorphism is a monomorphism -/
instance pullback.fst_of_mono {X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] [Mono g] :
    Mono (pullback.fst f g) :=
  PullbackCone.mono_fst_of_is_pullback_of_mono (limit.isLimit _)


/-- The pullback of a monomorphism is a monomorphism -/
instance pullback.snd_of_mono {X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] [Mono f] :
    Mono (pullback.snd f g) :=
  PullbackCone.mono_snd_of_is_pullback_of_mono (limit.isLimit _)


/-- The map `X ×[Z] Y ⟶ X × Y` is mono. -/
instance mono_pullback_to_prod {C : Type*} [Category C] {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z)
    [HasPullback f g] [HasBinaryProduct X Y] :
    Mono (prod.lift (pullback.fst f g) (pullback.snd f g)) :=
  ⟨fun {W} i₁ i₂ h => by
    /-
      C✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C✝
      W✝ X✝ Y✝ Z✝ : C✝
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝¹ : CategoryTheory.Limits.HasPullback f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
      W : C
      i₁ i₂ : Quiver.Hom W (CategoryTheory.Limits.pullback f g)
      h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.lift …
      ⊢ Eq i₁ i₂
    -/
    ext
      /-
        case h₀
        C✝ : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C✝
        W✝ X✝ Y✝ Z✝ : C✝
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
        W : C
        i₁ i₂ : Quiver.Hom W (CategoryTheory.Limits.pullback f g)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.lift …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.pullback.fs …
      -/
    · simpa using congrArg (fun f => f ≫ prod.fst) h
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C✝ : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C✝
        W✝ X✝ Y✝ Z✝ : C✝
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝¹ : CategoryTheory.Limits.HasPullback f g
        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
        W : C
        i₁ i₂ : Quiver.Hom W (CategoryTheory.Limits.pullback f g)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.lift …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.pullback.sn …
      -/
    · simpa using congrArg (fun f => f ≫ prod.snd) h⟩
      /-
        🎉 no goals
      -/


/-- The pullback of `f, g` is also the pullback of `f ≫ i, g ≫ i` for any mono `i`. -/
noncomputable def pullbackIsPullbackOfCompMono (f : X ⟶ W) (g : Y ⟶ W) (i : W ⟶ Z) [Mono i]
    [HasPullback f g] : IsLimit (PullbackCone.mk (pullback.fst f g) (pullback.snd f g)
      -- Porting note: following used to be _
      (show (pullback.fst f g) ≫ f ≫ i = (pullback.snd f g) ≫ g ≫ i from by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f : Quiver.Hom X W
          g : Quiver.Hom Y W
          i : Quiver.Hom W Z
          inst✝¹ : CategoryTheory.Mono i
          inst✝ : CategoryTheory.Limits.HasPullback f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst f …
        -/
        simp only [← Category.assoc]; rw [cancel_mono]; apply pullback.condition)) :=
                                                        /-
                                                          🎉 no goals
                                                        -/
  PullbackCone.isLimitOfCompMono f g i _ (limit.isLimit (cospan f g))


instance hasPullback_of_comp_mono (f : X ⟶ W) (g : Y ⟶ W) (i : W ⟶ Z) [Mono i] [HasPullback f g] :
    HasPullback (f ≫ i) (g ≫ i) :=
  ⟨⟨⟨_, pullbackIsPullbackOfCompMono f g i⟩⟩⟩


instance hasPullback_of_right_factors_mono : HasPullback i (f ≫ i) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.Limits.HasPullback i (CategoryTheory.CategoryStruct.comp f i)
  -/
  simpa only [Category.id_comp] using hasPullback_of_comp_mono (𝟙 Z) f i
  /-
    🎉 no goals
  -/


instance pullback_snd_iso_of_right_factors_mono :
    IsIso (pullback.snd i (f ≫ i)) := by
  #adaptation_note /-- nightly-testing 2024-04-01
  this could not be placed directly in the `show from` without `dsimp` -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd i (CategoryTheory.C …
  -/
  have := limit.isoLimitCone_hom_π ⟨_, pullbackIsPullbackOfCompMono (𝟙 _) f i⟩ WalkingCospan.right
  dsimp only [cospan_right, id_eq, eq_mpr_eq_cast, PullbackCone.mk_pt, PullbackCone.mk_π_app,
    Functor.const_obj_obj, cospan_one] at this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd i (CategoryTheory.C …
  -/
  convert (congrArg IsIso (show _ ≫ pullback.snd (𝟙 Z) f = _ from this)).mp inferInstance
    /-
      case h.e'_3.h.e'_6.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      i : Quiver.Hom Z W
      inst✝ : CategoryTheory.Mono i
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
      ⊢ Eq i (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Z …
    -/
  · exact (Category.id_comp _).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.e'_6.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      i : Quiver.Hom Z W
      inst✝ : CategoryTheory.Mono i
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
      e_3✝ : Eq (CategoryTheory.Limits.pullback i (CategoryTheory.CategoryStruct.com …
      ⊢ Eq i (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Z …
    -/
  · exact (Category.id_comp _).symm
    /-
      🎉 no goals
    -/


instance hasPullback_of_left_factors_mono : HasPullback (f ≫ i) i := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp f i) i
  -/
  simpa only [Category.id_comp] using hasPullback_of_comp_mono f (𝟙 Z) i
  /-
    🎉 no goals
  -/


instance pullback_snd_iso_of_left_factors_mono :
    IsIso (pullback.fst (f ≫ i) i) := by
  #adaptation_note /-- nightly-testing 2024-04-01
  this could not be placed directly in the `show from` without `dsimp` -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst (CategoryTheory.Cat …
  -/
  have := limit.isoLimitCone_hom_π ⟨_, pullbackIsPullbackOfCompMono f (𝟙 _) i⟩ WalkingCospan.left
  dsimp only [cospan_left, id_eq, eq_mpr_eq_cast, PullbackCone.mk_pt, PullbackCone.mk_π_app,
    Functor.const_obj_obj, cospan_one] at this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    i : Quiver.Hom Z W
    inst✝ : CategoryTheory.Mono i
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst (CategoryTheory.Cat …
  -/
  convert (congrArg IsIso (show _ ≫ pullback.fst f (𝟙 Z) = _ from this)).mp inferInstance
    /-
      case h.e'_3.h.e'_7.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      i : Quiver.Hom Z W
      inst✝ : CategoryTheory.Mono i
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
      ⊢ Eq i (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Z …
    -/
  · exact (Category.id_comp _).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.e'_7.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      i : Quiver.Hom Z W
      inst✝ : CategoryTheory.Mono i
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.iso …
      e_3✝ : Eq (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq i (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Z …
    -/
  · exact (Category.id_comp _).symm
    /-
      🎉 no goals
    -/


instance has_kernel_pair_of_mono : HasPullback f f :=
  ⟨⟨⟨_, PullbackCone.isLimitMkIdId f⟩⟩⟩


theorem PullbackCone.fst_eq_snd_of_mono_eq {f : X ⟶ Y} [Mono f] (t : PullbackCone f f) :
    t.fst = t.snd :=
  (cancel_mono f).1 t.condition


theorem fst_eq_snd_of_mono_eq : pullback.fst f f = pullback.snd f f :=
  PullbackCone.fst_eq_snd_of_mono_eq (getLimitCone (cospan f f)).cone


@[simp]
theorem pullbackSymmetry_hom_of_mono_eq : (pullbackSymmetry f f).hom = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (CategoryTheory.Limits.pullbackSymmetry f f).hom (CategoryTheory.Category …
  -/
  ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
    -/
  · simp [fst_eq_snd_of_mono_eq]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
    -/
  · simp [fst_eq_snd_of_mono_eq]
    /-
      🎉 no goals
    -/


variable {f} in
lemma PullbackCone.isIso_fst_of_mono_of_isLimit {t : PullbackCone f f} (ht : IsLimit t) :
    IsIso t.fst := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    t : CategoryTheory.Limits.PullbackCone f f
    ht : CategoryTheory.Limits.IsLimit t
    ⊢ CategoryTheory.IsIso t.fst
  -/
  refine ⟨⟨PullbackCone.IsLimit.lift ht (𝟙 _) (𝟙 _) (by simp), ?_, by simp⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    t : CategoryTheory.Limits.PullbackCone f f
    ht : CategoryTheory.Limits.IsLimit t
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t.fst (CategoryTheory.Limits.Pullback …
  -/
  apply PullbackCone.IsLimit.hom_ext ht
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      t : CategoryTheory.Limits.PullbackCone f f
      ht : CategoryTheory.Limits.IsLimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      t : CategoryTheory.Limits.PullbackCone f f
      ht : CategoryTheory.Limits.IsLimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp t …
    -/
  · simp [fst_eq_snd_of_mono_eq]
    /-
      🎉 no goals
    -/


variable {f} in
lemma PullbackCone.isIso_snd_of_mono_of_isLimit {t : PullbackCone f f} (ht : IsLimit t) :
    IsIso t.snd :=
  t.fst_eq_snd_of_mono_eq ▸ t.isIso_fst_of_mono_of_isLimit ht


instance isIso_fst_of_mono : IsIso (pullback.fst f f) :=
  PullbackCone.isIso_fst_of_mono_of_isLimit (getLimitCone (cospan f f)).isLimit


instance isIso_snd_of_mono : IsIso (pullback.snd f f) :=
  PullbackCone.isIso_snd_of_mono_of_isLimit (getLimitCone (cospan f f)).isLimit

theorem epi_inr_of_is_pushout_of_epi {t : PushoutCocone f g} (ht : IsColimit t) [Epi f] :
    Epi t.inr :=
                                             /-
                                               C : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                               X Y Z : C
                                               f : Quiver.Hom X Y
                                               g : Quiver.Hom X Z
                                               t : CategoryTheory.Limits.PushoutCocone f g
                                               ht : CategoryTheory.Limits.IsColimit t
                                               inst✝ : CategoryTheory.Epi f
                                               W : C
                                               h k : Quiver.Hom t.pt W
                                               i : Eq (CategoryTheory.CategoryStruct.comp t.inr h) (CategoryTheory.CategorySt …
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp t.inl h) (CategoryTheory.CategoryStru …
                                             -/
  ⟨fun {W} h k i => IsColimit.hom_ext ht (by simp [← cancel_epi f, t.condition_assoc, i]) i⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem epi_inl_of_is_pushout_of_epi {t : PushoutCocone f g} (ht : IsColimit t) [Epi g] :
    Epi t.inl :=
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 X Y Z : C
                                                 f : Quiver.Hom X Y
                                                 g : Quiver.Hom X Z
                                                 t : CategoryTheory.Limits.PushoutCocone f g
                                                 ht : CategoryTheory.Limits.IsColimit t
                                                 inst✝ : CategoryTheory.Epi g
                                                 W : C
                                                 h k : Quiver.Hom t.pt W
                                                 i : Eq (CategoryTheory.CategoryStruct.comp t.inl h) (CategoryTheory.CategorySt …
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp t.inr h) (CategoryTheory.CategoryStru …
                                               -/
  ⟨fun {W} h k i => IsColimit.hom_ext ht i (by simp [← cancel_epi g, ← t.condition_assoc, i])⟩
                                               /-
                                                 🎉 no goals
                                               -/


/--
The pushout cocone `(𝟙 X, 𝟙 X)` for the pair `(f, f)` is a colimit if `f` is an epi. The converse is
shown in `epi_of_isColimit_mk_id_id`.
-/
def isColimitMkIdId (f : X ⟶ Y) [Epi f] : IsColimit (mk (𝟙 Y) (𝟙 Y) rfl : PushoutCocone f f) :=
  IsColimit.mk _ (fun s => s.inl) (fun _ => Category.id_comp _)
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   W X Y Z : C
                   f✝ : Quiver.Hom X Y
                   g : Quiver.Hom X Z
                   f : Quiver.Hom X Y
                   inst✝ : CategoryTheory.Epi f
                   s : CategoryTheory.Limits.PushoutCocone f f
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
                 -/
    (fun s => by rw [← cancel_epi f, Category.id_comp, s.condition]) fun s m m₁ _ => by
                 /-
                   🎉 no goals
                 -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f✝ : Quiver.Hom X Y
      g : Quiver.Hom X Z
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Epi f
      s : CategoryTheory.Limits.PushoutCocone f f
      m : Quiver.Hom Y s.pt
      m₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
      x✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
      ⊢ Eq m ((fun s => s.inl) s)
    -/
    simpa using m₁
    /-
      🎉 no goals
    -/


/-- `f` is an epi if the pushout cocone `(𝟙 X, 𝟙 X)` is a colimit for the pair `(f, f)`.
The converse is given in `PushoutCocone.isColimitMkIdId`.
-/
theorem epi_of_isColimitMkIdId (f : X ⟶ Y)
    (t : IsColimit (mk (𝟙 Y) (𝟙 Y) rfl : PushoutCocone f f)) : Epi f :=
  ⟨fun {Z} g h eq => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (C …
      Z : C
      g h : Quiver.Hom Y Z
      eq : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    rcases PushoutCocone.IsColimit.desc' t _ _ eq with ⟨_, rfl, rfl⟩
    /-
      case mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (C …
      Z : C
      val✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk (CategoryTheory.Cate …
      eq : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.c …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


/-- Suppose `f` and `g` are two morphisms with a common domain and `s` is a colimit cocone over the
    diagram formed by `f` and `g`. Suppose `f` and `g` both factor through an epimorphism `h` via
    `x` and `y`, respectively. Then `s` is also a colimit cocone over the diagram formed by `x` and
    `y`. -/
def isColimitOfFactors (f : X ⟶ Y) (g : X ⟶ Z) (h : X ⟶ W) [Epi h] (x : W ⟶ Y) (y : W ⟶ Z)
    (hhx : h ≫ x = f) (hhy : h ≫ y = g) (s : PushoutCocone f g) (hs : IsColimit s) :
    have reassoc₁ : h ≫ x ≫ inl s = f ≫ inl s := by  -- Porting note: working around reassoc
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f✝ : Quiver.Hom X Y
        g✝ : Quiver.Hom X Z
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom X W
        inst✝ : CategoryTheory.Epi h
        x : Quiver.Hom W Y
        y : Quiver.Hom W Z
        hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
        hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
        s : CategoryTheory.Limits.PushoutCocone f g
        hs : CategoryTheory.Limits.IsColimit s
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
      -/
      rw [← Category.assoc]; apply congrArg (· ≫ inl s) hhx
                             /-
                               🎉 no goals
                             -/
    have reassoc₂ : h ≫ y ≫ inr s = g ≫ inr s := by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        W X Y Z : C
        f✝ : Quiver.Hom X Y
        g✝ : Quiver.Hom X Z
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        h : Quiver.Hom X W
        inst✝ : CategoryTheory.Epi h
        x : Quiver.Hom W Y
        y : Quiver.Hom W Z
        hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
        hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
        s : CategoryTheory.Limits.PushoutCocone f g
        hs : CategoryTheory.Limits.IsColimit s
        reassoc₁ : Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
      -/
      rw [← Category.assoc]; apply congrArg (· ≫ inr s) hhy
                             /-
                               🎉 no goals
                             -/
    IsColimit (PushoutCocone.mk _ _ (show x ≫ s.inl = y ≫ s.inr from
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   W X Y Z : C
                                   f✝ : Quiver.Hom X Y
                                   g✝ : Quiver.Hom X Z
                                   f : Quiver.Hom X Y
                                   g : Quiver.Hom X Z
                                   h : Quiver.Hom X W
                                   inst✝ : CategoryTheory.Epi h
                                   x : Quiver.Hom W Y
                                   y : Quiver.Hom W Z
                                   hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
                                   hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
                                   s : CategoryTheory.Limits.PushoutCocone f g
                                   hs : CategoryTheory.Limits.IsColimit s
                                   reassoc₁ : Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategorySt …
                                   reassoc₂ : Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategorySt …
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
                                 -/
          (cancel_epi h).1 <| by rw [reassoc₁, reassoc₂, s.condition])) :=
                                 /-
                                   🎉 no goals
                                 -/
  PushoutCocone.isColimitAux' _ fun t => ⟨hs.desc (PushoutCocone.mk t.inl t.inr <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f✝ : Quiver.Hom X Y
      g✝ : Quiver.Hom X Z
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      h : Quiver.Hom X W
      inst✝ : CategoryTheory.Epi h
      x : Quiver.Hom W Y
      y : Quiver.Hom W Z
      hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
      hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
      s : CategoryTheory.Limits.PushoutCocone f g
      hs : CategoryTheory.Limits.IsColimit s
      t : CategoryTheory.Limits.PushoutCocone x y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f t.inl) (CategoryTheory.CategoryStru …
    -/
    rw [← hhx, ← hhy, Category.assoc, Category.assoc, t.condition]),
    /-
      🎉 no goals
    -/
      ⟨hs.fac _ WalkingSpan.left, hs.fac _ WalkingSpan.right, fun hr hr' => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          f✝ : Quiver.Hom X Y
          g✝ : Quiver.Hom X Z
          f : Quiver.Hom X Y
          g : Quiver.Hom X Z
          h : Quiver.Hom X W
          inst✝ : CategoryTheory.Epi h
          x : Quiver.Hom W Y
          y : Quiver.Hom W Z
          hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
          hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
          s : CategoryTheory.Limits.PushoutCocone f g
          hs : CategoryTheory.Limits.IsColimit s
          t : CategoryTheory.Limits.PushoutCocone x y
          m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
          hr : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
          hr' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
          ⊢ Eq m✝ (hs.desc (CategoryTheory.Limits.PushoutCocone.mk t.inl t.inr ⋯))
        -/
        apply PushoutCocone.IsColimit.hom_ext hs
          /-
            case h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
            hr' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) (CategoryTheory.CategoryStr …
          -/
        · simp only [PushoutCocone.mk_inl, PushoutCocone.mk_inr] at hr hr' ⊢
          /-
            case h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) (CategoryTheory.CategoryStr …
          -/
          simp only [hr, hr']
          /-
            case h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq t.inl (CategoryTheory.CategoryStruct.comp s.inl (hs.desc (CategoryTheory. …
          -/
          symm
          /-
            case h₀
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inl (hs.desc (CategoryTheory.Limits …
          -/
          exact hs.fac _ WalkingSpan.left
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoco …
            hr' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) (CategoryTheory.CategoryStr …
          -/
        · simp only [PushoutCocone.mk_inl, PushoutCocone.mk_inr] at hr hr' ⊢
          /-
            case h₁
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) (CategoryTheory.CategoryStr …
          -/
          simp only [hr, hr']
          /-
            case h₁
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq t.inr (CategoryTheory.CategoryStruct.comp s.inr (hs.desc (CategoryTheory. …
          -/
          symm
          /-
            case h₁
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom X W
            inst✝ : CategoryTheory.Epi h
            x : Quiver.Hom W Y
            y : Quiver.Hom W Z
            hhx : Eq (CategoryTheory.CategoryStruct.comp h x) f
            hhy : Eq (CategoryTheory.CategoryStruct.comp h y) g
            s : CategoryTheory.Limits.PushoutCocone f g
            hs : CategoryTheory.Limits.IsColimit s
            t : CategoryTheory.Limits.PushoutCocone x y
            m✝ : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s.inl s.inr ⋯).pt t.pt
            hr : Eq (CategoryTheory.CategoryStruct.comp s.inl m✝) t.inl
            hr' : Eq (CategoryTheory.CategoryStruct.comp s.inr m✝) t.inr
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.inr (hs.desc (CategoryTheory.Limits …
          -/
          exact hs.fac _ WalkingSpan.right⟩⟩
          /-
            🎉 no goals
          -/


/-- If `W` is the pushout of `f, g`,
it is also the pushout of `h ≫ f, h ≫ g` for any epi `h`. -/
def isColimitOfEpiComp (f : X ⟶ Y) (g : X ⟶ Z) (h : W ⟶ X) [Epi h] (s : PushoutCocone f g)
    (H : IsColimit s) :
    IsColimit
      (PushoutCocone.mk _ _
        (show (h ≫ f) ≫ s.inl = (h ≫ g) ≫ s.inr by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            W X Y Z : C
            f✝ : Quiver.Hom X Y
            g✝ : Quiver.Hom X Z
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            h : Quiver.Hom W X
            inst✝ : CategoryTheory.Epi h
            s : CategoryTheory.Limits.PushoutCocone f g
            H : CategoryTheory.Limits.IsColimit s
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
          -/
          rw [Category.assoc, Category.assoc, s.condition])) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Y
    g✝ : Quiver.Hom X Z
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    s : CategoryTheory.Limits.PushoutCocone f g
    H : CategoryTheory.Limits.IsColimit s
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk s.in …
  -/
  apply PushoutCocone.isColimitAux'
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Y
    g✝ : Quiver.Hom X Z
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    s : CategoryTheory.Limits.PushoutCocone f g
    H : CategoryTheory.Limits.IsColimit s
    ⊢ (s_1 : CategoryTheory.Limits.PushoutCocone (CategoryTheory.CategoryStruct.co …
  -/
  intro s
  rcases PushoutCocone.IsColimit.desc' H s.inl s.inr
      ((cancel_epi h).mp (by simpa using s.condition)) with
    ⟨l, h₁, h₂⟩
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Y
    g✝ : Quiver.Hom X Z
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    s✝ : CategoryTheory.Limits.PushoutCocone f g
    H : CategoryTheory.Limits.IsColimit s✝
    s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.CategoryStruct.comp h  …
    l : Quiver.Hom s✝.pt s.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp s✝.inl l) s.inl
    h₂ : Eq (CategoryTheory.CategoryStruct.comp s✝.inr l) s.inr
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
  -/
  refine ⟨l, h₁, h₂, ?_⟩
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Y
    g✝ : Quiver.Hom X Z
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    s✝ : CategoryTheory.Limits.PushoutCocone f g
    H : CategoryTheory.Limits.IsColimit s✝
    s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.CategoryStruct.comp h  …
    l : Quiver.Hom s✝.pt s.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp s✝.inl l) s.inl
    h₂ : Eq (CategoryTheory.CategoryStruct.comp s✝.inr l) s.inr
    ⊢ ∀ {m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s✝.inl s✝.inr ⋯).p …
  -/
  intro m hm₁ hm₂
  /-
    case create.mk.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Y
    g✝ : Quiver.Hom X Z
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    s✝ : CategoryTheory.Limits.PushoutCocone f g
    H : CategoryTheory.Limits.IsColimit s✝
    s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.CategoryStruct.comp h  …
    l : Quiver.Hom s✝.pt s.pt
    h₁ : Eq (CategoryTheory.CategoryStruct.comp s✝.inl l) s.inl
    h₂ : Eq (CategoryTheory.CategoryStruct.comp s✝.inr l) s.inr
    m : Quiver.Hom (CategoryTheory.Limits.PushoutCocone.mk s✝.inl s✝.inr ⋯).pt s.pt
    hm₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
    hm₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCoc …
    ⊢ Eq m l
  -/
  exact (PushoutCocone.IsColimit.hom_ext H (hm₁.trans h₁.symm) (hm₂.trans h₂.symm) : _)
  /-
    🎉 no goals
  -/


/-- The pushout of an epimorphism is an epimorphism -/
instance pushout.inl_of_epi {X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] [Epi g] :
    Epi (pushout.inl f g) :=
  PushoutCocone.epi_inl_of_is_pushout_of_epi (colimit.isColimit _)


/-- The pushout of an epimorphism is an epimorphism -/
instance pushout.inr_of_epi {X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] [Epi f] :
    Epi (pushout.inr _ _ : Z ⟶ pushout f g) :=
  PushoutCocone.epi_inr_of_is_pushout_of_epi (colimit.isColimit _)


/-- The map `X ⨿ Y ⟶ X ⨿[Z] Y` is epi. -/
instance epi_coprod_to_pushout {C : Type*} [Category C] {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z)
    [HasPushout f g] [HasBinaryCoproduct Y Z] :
    Epi (coprod.desc (pushout.inl f g) (pushout.inr f g)) :=
  ⟨fun {W} i₁ i₂ h => by
    /-
      C✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C✝
      W✝ X✝ Y✝ Z✝ : C✝
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝¹ : CategoryTheory.Limits.HasPushout f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
      W : C
      i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.pushout f g) W
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc  …
      ⊢ Eq i₁ i₂
    -/
    ext
      /-
        case h₀
        C✝ : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C✝
        W✝ X✝ Y✝ Z✝ : C✝
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPushout f g
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        W : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.pushout f g) W
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
      -/
    · simpa using congrArg (fun f => coprod.inl ≫ f) h
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C✝ : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C✝
        W✝ X✝ Y✝ Z✝ : C✝
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom X Z
        inst✝¹ : CategoryTheory.Limits.HasPushout f g
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        W : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.pushout f g) W
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
      -/
    · simpa using congrArg (fun f => coprod.inr ≫ f) h⟩
      /-
        🎉 no goals
      -/


/-- The pushout of `f, g` is also the pullback of `h ≫ f, h ≫ g` for any epi `h`. -/
noncomputable def pushoutIsPushoutOfEpiComp (f : X ⟶ Y) (g : X ⟶ Z) (h : W ⟶ X) [Epi h]
    [HasPushout f g] : IsColimit (PushoutCocone.mk (pushout.inl f g) (pushout.inr f g)
    (show (h ≫ f) ≫ pushout.inl f g = (h ≫ g) ≫ pushout.inr f g from by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      h : Quiver.Hom W X
      inst✝¹ : CategoryTheory.Epi h
      inst✝ : CategoryTheory.Limits.HasPushout f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
    -/
    simp only [Category.assoc]; rw [cancel_epi]; exact pushout.condition)) :=
                                                 /-
                                                   🎉 no goals
                                                 -/
  PushoutCocone.isColimitOfEpiComp f g h _ (colimit.isColimit (span f g))


instance hasPushout_of_epi_comp (f : X ⟶ Y) (g : X ⟶ Z) (h : W ⟶ X) [Epi h] [HasPushout f g] :
    HasPushout (h ≫ f) (h ≫ g) :=
  ⟨⟨⟨_, pushoutIsPushoutOfEpiComp f g h⟩⟩⟩


instance hasPushout_of_right_factors_epi : HasPushout h (h ≫ f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    ⊢ CategoryTheory.Limits.HasPushout h (CategoryTheory.CategoryStruct.comp h f)
  -/
  simpa only [Category.comp_id] using hasPushout_of_epi_comp (𝟙 X) f h
  /-
    🎉 no goals
  -/


instance pushout_inr_iso_of_right_factors_epi :
    IsIso (pushout.inr _ _ : _ ⟶ pushout h (h ≫ f)) := by
  convert (congrArg IsIso (show pushout.inr _ _ ≫ _ = _ from colimit.isoColimitCocone_ι_inv
    ⟨_, pushoutIsPushoutOfEpiComp (𝟙 _) f h⟩ WalkingSpan.right)).mp
    inferInstance
    /-
      case h.e'_4.h.e'_6.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      h : Quiver.Hom W X
      inst✝ : CategoryTheory.Epi h
      ⊢ Eq h (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.id …
    -/
  · apply (Category.comp_id _).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.e'_6.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f : Quiver.Hom X Z
      h : Quiver.Hom W X
      inst✝ : CategoryTheory.Epi h
      e_3✝ : Eq Z ((CategoryTheory.Limits.span (CategoryTheory.CategoryStruct.comp h …
      e_4✝ : Eq (CategoryTheory.Limits.pushout h (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq h (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.id …
    -/
  · apply (Category.comp_id _).symm
    /-
      🎉 no goals
    -/


instance hasPushout_of_left_factors_epi (f : X ⟶ Y) : HasPushout (h ≫ f) h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f✝ : Quiver.Hom X Z
    h : Quiver.Hom W X
    inst✝ : CategoryTheory.Epi h
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp h f) h
  -/
  simpa only [Category.comp_id] using hasPushout_of_epi_comp f (𝟙 X) h
  /-
    🎉 no goals
  -/


instance pushout_inl_iso_of_left_factors_epi (f : X ⟶ Y) :
    IsIso (pushout.inl _ _ : _ ⟶ pushout (h ≫ f) h) := by
  convert (congrArg IsIso (show pushout.inl _ _ ≫ _ = _ from colimit.isoColimitCocone_ι_inv
    ⟨_, pushoutIsPushoutOfEpiComp f (𝟙 _) h⟩ WalkingSpan.left)).mp
        inferInstance
    /-
      case h.e'_4.h.e'_7.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f✝ : Quiver.Hom X Z
      h : Quiver.Hom W X
      inst✝ : CategoryTheory.Epi h
      f : Quiver.Hom X Y
      ⊢ Eq h (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.id …
    -/
  · exact (Category.comp_id _).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.e'_7.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      W X Y Z : C
      f✝ : Quiver.Hom X Z
      h : Quiver.Hom W X
      inst✝ : CategoryTheory.Epi h
      f : Quiver.Hom X Y
      e_3✝ : Eq Y ((CategoryTheory.Limits.span (CategoryTheory.CategoryStruct.comp h …
      e_4✝ : Eq (CategoryTheory.Limits.pushout (CategoryTheory.CategoryStruct.comp h …
      ⊢ Eq h (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.id …
    -/
  · exact (Category.comp_id _).symm
    /-
      🎉 no goals
    -/


instance has_cokernel_pair_of_epi : HasPushout f f :=
  ⟨⟨⟨_, PushoutCocone.isColimitMkIdId f⟩⟩⟩


theorem PushoutCocone.inl_eq_inr_of_epi_eq {f : X ⟶ Y} [Epi f] (t : PushoutCocone f f) :
    t.inl = t.inr :=
  (cancel_epi f).1 t.condition


theorem inl_eq_inr_of_epi_eq : pushout.inl f f = pushout.inr f f :=
  PushoutCocone.inl_eq_inr_of_epi_eq (getColimitCocone (span f f)).cocone


@[simp]
theorem pullback_symmetry_hom_of_epi_eq : (pushoutSymmetry f f).hom = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    ⊢ Eq (CategoryTheory.Limits.pushoutSymmetry f f).hom (CategoryTheory.CategoryS …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [inl_eq_inr_of_epi_eq]
          /-
            🎉 no goals
          -/


variable {f} in
lemma PushoutCocone.isIso_inl_of_epi_of_isColimit {t : PushoutCocone f f} (ht : IsColimit t) :
    IsIso t.inl := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    t : CategoryTheory.Limits.PushoutCocone f f
    ht : CategoryTheory.Limits.IsColimit t
    ⊢ CategoryTheory.IsIso t.inl
  -/
  refine ⟨⟨PushoutCocone.IsColimit.desc ht (𝟙 _) (𝟙 _) (by simp), by simp, ?_⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    t : CategoryTheory.Limits.PushoutCocone f f
    ht : CategoryTheory.Limits.IsColimit t
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PushoutCocone. …
  -/
  apply PushoutCocone.IsColimit.hom_ext ht
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Epi f
      t : CategoryTheory.Limits.PushoutCocone f f
      ht : CategoryTheory.Limits.IsColimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t.inl (CategoryTheory.CategoryStruct. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Epi f
      t : CategoryTheory.Limits.PushoutCocone f f
      ht : CategoryTheory.Limits.IsColimit t
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t.inr (CategoryTheory.CategoryStruct. …
    -/
  · simp [inl_eq_inr_of_epi_eq]
    /-
      🎉 no goals
    -/


variable {f} in
lemma PushoutCocone.isIso_inr_of_epi_of_isColimit {t : PushoutCocone f f} (ht : IsColimit t) :
    IsIso t.inr :=
  t.inl_eq_inr_of_epi_eq ▸ t.isIso_inl_of_epi_of_isColimit ht


instance isIso_inl_of_epi : IsIso (pushout.inl f f) :=
  PushoutCocone.isIso_inl_of_epi_of_isColimit (getColimitCocone (span f f)).isColimit


instance isIso_inr_of_epi : IsIso (pushout.inr f f) :=
  PushoutCocone.isIso_inr_of_epi_of_isColimit (getColimitCocone (span f f)).isColimit


