variable (F) in
/-- A directed system is a functor from a category (directed poset) to another category. -/
class DirectedSystem (f : ∀ ⦃i j⦄, i ≤ j → F i → F j) : Prop where
  map_self ⦃i⦄ (x : F i) : f le_rfl x = x
  map_map ⦃k j i⦄ (hij : i ≤ j) (hjk : j ≤ k) (x : F i) : f hjk (f hij x) = f (hij.trans hjk) x


/-- A copy of `DirectedSystem.map_self` specialized to FunLike, as otherwise the
`fun i j h ↦ f i j h` can confuse the simplifier. -/
theorem DirectedSystem.map_self' ⦃i⦄ (x) : f i i le_rfl x = x :=
  DirectedSystem.map_self (f := (f · · ·)) x


/-- A copy of `DirectedSystem.map_map` specialized to FunLike, as otherwise the
`fun i j h ↦ f i j h` can confuse the simplifier. -/
theorem DirectedSystem.map_map' ⦃i j k⦄ (hij hjk x) :
    f j k hjk (f i j hij x) = f i k (hij.trans hjk) x :=
  DirectedSystem.map_map (f := (f · · ·)) hij hjk x


/-- The setoid on the sigma type defining the direct limit. -/
def setoid : Setoid (Σ i, F i) where
  r x y := ∃ᵉ (i) (hx : x.1 ≤ i) (hy : y.1 ≤ i), f _ _ hx x.2 = f _ _ hy y.2
  iseqv := ⟨fun x ↦ ⟨x.1, le_rfl, le_rfl, rfl⟩, fun ⟨i, hx, hy, eq⟩ ↦ ⟨i, hy, hx, eq.symm⟩,
    fun ⟨j, hx, _, jeq⟩ ⟨k, _, hz, keq⟩ ↦
      have ⟨i, hji, hki⟩ := exists_ge_ge j k
      ⟨i, hx.trans hji, hz.trans hki, by
        /-
          ι : Type u_1
          inst✝⁷ : Preorder ι
          F₁ : ι → Type u_2
          F₂ : ι → Type u_3
          F : ι → Type u_4
          X : ι → Type u_5
          T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
          f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
          inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
          inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
          T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
          f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
          inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
          inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
          T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
          f : (i j : ι) → (h : LE.le i j) → T h
          inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
          inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          x✝² y✝ z✝ : Sigma fun i => F i
          x✝¹ : Exists fun i => Exists fun hx => Exists fun hy => Eq ((f x✝².fst i hx) x …
          x✝ : Exists fun i => Exists fun hx => Exists fun hy => Eq ((f y✝.fst i hx) y✝. …
          j : ι
          hx : LE.le x✝².fst j
          w✝¹ : LE.le y✝.fst j
          jeq : Eq ((f x✝².fst j hx) x✝².snd) ((f y✝.fst j w✝¹) y✝.snd)
          k : ι
          w✝ : LE.le y✝.fst k
          hz : LE.le z✝.fst k
          keq : Eq ((f y✝.fst k w✝) y✝.snd) ((f z✝.fst k hz) z✝.snd)
          i : ι
          hji : LE.le j i
          hki : LE.le k i
          ⊢ Eq ((f x✝².fst i ⋯) x✝².snd) ((f z✝.fst i ⋯) z✝.snd)
        -/
        rw [← map_map' _ hx hji, ← map_map' _ hz hki, jeq, ← keq, map_map', map_map']⟩⟩
        /-
          🎉 no goals
        -/


theorem r_of_le (x : Σ i, F i) (i : ι) (h : x.1 ≤ i) : (setoid f).r x ⟨i, f _ _ h x.2⟩ :=
  ⟨i, h, le_rfl, (map_map' _ _ _ _).symm⟩


variable (F) in
/-- The direct limit of a directed system. -/
abbrev _root_.DirectLimit : Type _ := Quotient (setoid f)


variable {f} in
theorem eq_of_le (x : Σ i, F i) (i : ι) (h : x.1 ≤ i) :
    (⟦x⟧ : DirectLimit F f) = ⟦⟨i, f _ _ h x.2⟩⟧ :=
  Quotient.sound (r_of_le _ x i h)


@[elab_as_elim] protected theorem induction {C : DirectLimit F f → Prop}
    (ih : ∀ i x, C ⟦⟨i, x⟩⟧) (x : DirectLimit F f) : C x :=
  Quotient.ind (fun _ ↦ ih _ _) x


                                                                       /-
                                                                         ι : Type u_1
                                                                         inst✝³ : Preorder ι
                                                                         F : ι → Type u_4
                                                                         T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
                                                                         f : (i j : ι) → (h : LE.le i j) → T h
                                                                         inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
                                                                         inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                                                         inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                                         z : DirectLimit F f
                                                                         ⊢ Exists fun i => Exists fun x => Eq z (Quotient.mk (DirectLimit.setoid f) ⟨i, …
                                                                       -/
theorem exists_eq_mk (z : DirectLimit F f) : ∃ i x, z = ⟦⟨i, x⟩⟧ := by rcases z; exact ⟨_, _, rfl⟩
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem exists_eq_mk₂ (z w : DirectLimit F f) : ∃ i x y, z = ⟦⟨i, x⟩⟧ ∧ w = ⟦⟨i, y⟩⟧ :=
  z.inductionOn₂ w fun x y ↦
    have ⟨i, hxi, hyi⟩ := exists_ge_ge x.1 y.1
    ⟨i, _, _, eq_of_le x i hxi, eq_of_le y i hyi⟩


theorem exists_eq_mk₃ (w u v : DirectLimit F f) :
    ∃ i x y z, w = ⟦⟨i, x⟩⟧ ∧ u = ⟦⟨i, y⟩⟧ ∧ v = ⟦⟨i, z⟩⟧ :=
  w.inductionOn₃ u v fun x y z ↦
    have ⟨i, hxi, hyi, hzi⟩ := directed_of₃ (· ≤ ·) x.1 y.1 z.1
    ⟨i, _, _, _, eq_of_le x i hxi, eq_of_le y i hyi, eq_of_le z i hzi⟩


@[elab_as_elim] protected theorem induction₂ {C : DirectLimit F f → DirectLimit F f → Prop}
    (ih : ∀ i x y, C ⟦⟨i, x⟩⟧ ⟦⟨i, y⟩⟧) (x y : DirectLimit F f) : C x y := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    F : ι → Type u_4
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : DirectLimit F f → DirectLimit F f → Prop
    ih : ∀ (i : ι) (x y : F i), C (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Quo …
    x y : DirectLimit F f
    ⊢ C x y
  -/
  obtain ⟨_, _, _, rfl, rfl⟩ := exists_eq_mk₂ f x y; apply ih
                                                     /-
                                                       🎉 no goals
                                                     -/


@[elab_as_elim] protected theorem induction₃
    {C : DirectLimit F f → DirectLimit F f → DirectLimit F f → Prop}
    (ih : ∀ i x y z, C ⟦⟨i, x⟩⟧ ⟦⟨i, y⟩⟧ ⟦⟨i, z⟩⟧) (x y z : DirectLimit F f) : C x y z := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    F : ι → Type u_4
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : DirectLimit F f → DirectLimit F f → DirectLimit F f → Prop
    ih : ∀ (i : ι) (x y z : F i), C (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Q …
    x y z : DirectLimit F f
    ⊢ C x y z
  -/
  obtain ⟨_, _, _, _, rfl, rfl, rfl⟩ := exists_eq_mk₃ f x y z; apply ih
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem mk_injective (h : ∀ i j hij, Function.Injective (f i j hij)) (i) :
    Function.Injective fun x ↦ (⟦⟨i, x⟩⟧ : DirectLimit F f) :=
  fun _ _ eq ↦ have ⟨_, _, _, eq⟩ := Quotient.eq.mp eq; h _ _ _ eq


/-- "Nullary map" to construct an element in the direct limit. -/
noncomputable def map₀ : DirectLimit F f := ⟦⟨arbitrary ι, ih _⟩⟧


theorem map₀_def (compat : ∀ i j h, f i j h (ih i) = ih j) (i) : map₀ f ih = ⟦⟨i, ih i⟩⟧ :=
  have ⟨j, hcj, hij⟩ := exists_ge_ge (arbitrary ι) i
  Quotient.sound ⟨j, hcj, hij, (compat ..).trans (compat ..).symm⟩


/-- To define a function from the direct limit, it suffices to provide one function from each
component subject to a compatibility condition. -/
protected def lift (z : DirectLimit F f) : C :=
  z.recOn (fun x ↦ ih x.1 x.2) fun x y ⟨k, hxk, hyk, eq⟩ ↦ by
    /-
      ι : Type u_1
      inst✝⁷ : Preorder ι
      F₁ : ι → Type u_2
      F₂ : ι → Type u_3
      F : ι → Type u_4
      X : ι → Type u_5
      T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
      f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
      inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
      inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
      T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
      f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
      inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
      inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
      T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
      f : (i j : ι) → (h : LE.le i j) → T h
      inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
      inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      C : Sort u_9
      ih : (i : ι) → F i → C
      compat : ∀ (i j : ι) (h : LE.le i j) (x : F i), Eq (ih i x) (ih j ((f i j h) x))
      z : DirectLimit F f
      x y : Sigma fun i => F i
      x✝ : HasEquiv.Equiv x y
      k : ι
      hxk : LE.le x.fst k
      hyk : LE.le y.fst k
      eq : Eq ((f x.fst k hxk) x.snd) ((f y.fst k hyk) y.snd)
      ⊢ Eq (Eq.ndrec (ih x.fst x.snd) ⋯) (ih y.fst y.snd)
    -/
    simp_rw [eq_rec_constant, compat _ _ hxk, compat _ _ hyk, eq]
    /-
      🎉 no goals
    -/


theorem lift_def (x) : DirectLimit.lift f ih compat ⟦x⟧ = ih x.1 x.2 := rfl


theorem lift_injective (h : ∀ i, Function.Injective (ih i)) :
    Function.Injective (DirectLimit.lift f ih compat) :=
                                             /-
                                               ι : Type u_1
                                               inst✝³ : Preorder ι
                                               F : ι → Type u_4
                                               T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
                                               f : (i j : ι) → (h : LE.le i j) → T h
                                               inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
                                               inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
                                               inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                               C : Sort u_9
                                               ih : (i : ι) → F i → C
                                               compat : ∀ (i j : ι) (h : LE.le i j) (x : F i), Eq (ih i x) (ih j ((f i j h) x))
                                               h : ∀ (i : ι), Function.Injective (ih i)
                                               i : ι
                                               x y : F i
                                               eq : Eq (DirectLimit.lift f ih compat (Quotient.mk (DirectLimit.setoid f) ⟨i,  …
                                               ⊢ Eq (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Quotient.mk (DirectLimit.set …
                                             -/
  DirectLimit.induction₂ _ fun i x y eq ↦ by simp_rw [lift_def] at eq; rw [h i eq]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- To define a function from the direct limit, it suffices to provide one function from each
component subject to a compatibility condition. -/
def map (z : DirectLimit F₁ f₁) : DirectLimit F₂ f₂ :=
  z.lift _ (fun i x ↦ ⟦⟨i, ih i x⟩⟧) fun j k h x ↦ Quotient.sound <|
    have ⟨i, hji, hki⟩ := exists_ge_ge j k
                     /-
                       ι : Type u_1
                       inst✝⁷ : Preorder ι
                       F₁ : ι → Type u_2
                       F₂ : ι → Type u_3
                       F : ι → Type u_4
                       X : ι → Type u_5
                       T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
                       f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
                       inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
                       inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
                       T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
                       f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
                       inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
                       inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
                       T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
                       f : (i j : ι) → (h : LE.le i j) → T h
                       inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
                       inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
                       inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                       ih : (i : ι) → F₁ i → F₂ i
                       compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i), Eq ((f₂ i j h) (ih i x)) (ih  …
                       z : DirectLimit F₁ f₁
                       j k : ι
                       h : LE.le j k
                       x : F₁ j
                       i : ι
                       hji : LE.le j i
                       hki : LE.le k i
                       ⊢ Eq ((f₂ ⟨j, ih j x⟩.fst i hji) ⟨j, ih j x⟩.snd) ((f₂ ⟨k, ih k ((f₁ j k h) x) …
                     -/
    ⟨i, hji, hki, by simp_rw [compat, map_map']⟩
                     /-
                       🎉 no goals
                     -/


theorem map_def (x) : map f₁ f₂ ih compat ⟦x⟧ = ⟦⟨x.1, ih x.1 x.2⟩⟧ := rfl


private noncomputable def lift₂Aux (z : Σ i, F₁ i) (w : Σ i, F₂ i) :
    {x : C // ∀ i (hzi : z.1 ≤ i) (hwi : w.1 ≤ i), x = ih i (f₁ _ _ hzi z.2) (f₂ _ _ hwi w.2)} := by
  /-
    ι : Type u_1
    inst✝⁷ : Preorder ι
    F₁ : ι → Type u_2
    F₂ : ι → Type u_3
    F : ι → Type u_4
    X : ι → Type u_5
    T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
    f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
    inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
    inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
    T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
    f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
    inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
    inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : Sort u_9
    ih : (i : ι) → F₁ i → F₂ i → C
    compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
    z : Sigma fun i => F₁ i
    w : Sigma fun i => F₂ i
    ⊢ Subtype fun x => ∀ (i : ι) (hzi : LE.le z.fst i) (hwi : LE.le w.fst i), Eq x …
  -/
  choose j hzj hwj using exists_ge_ge z.1 w.1
  /-
    ι : Type u_1
    inst✝⁷ : Preorder ι
    F₁ : ι → Type u_2
    F₂ : ι → Type u_3
    F : ι → Type u_4
    X : ι → Type u_5
    T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
    f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
    inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
    inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
    T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
    f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
    inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
    inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : Sort u_9
    ih : (i : ι) → F₁ i → F₂ i → C
    compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
    z : Sigma fun i => F₁ i
    w : Sigma fun i => F₂ i
    j : ι
    hzj : LE.le z.fst j
    hwj : LE.le w.fst j
    ⊢ Subtype fun x => ∀ (i : ι) (hzi : LE.le z.fst i) (hwi : LE.le w.fst i), Eq x …
  -/
  refine ⟨ih j (f₁ _ _ hzj z.2) (f₂ _ _ hwj w.2), fun k hzk hwk ↦ ?_⟩
  /-
    ι : Type u_1
    inst✝⁷ : Preorder ι
    F₁ : ι → Type u_2
    F₂ : ι → Type u_3
    F : ι → Type u_4
    X : ι → Type u_5
    T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
    f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
    inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
    inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
    T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
    f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
    inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
    inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : Sort u_9
    ih : (i : ι) → F₁ i → F₂ i → C
    compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
    z : Sigma fun i => F₁ i
    w : Sigma fun i => F₂ i
    j : ι
    hzj : LE.le z.fst j
    hwj : LE.le w.fst j
    k : ι
    hzk : LE.le z.fst k
    hwk : LE.le w.fst k
    ⊢ Eq (ih j ((f₁ z.fst j hzj) z.snd) ((f₂ w.fst j hwj) w.snd)) (ih k ((f₁ z.fst …
  -/
  have ⟨i, hji, hki⟩ := exists_ge_ge j k
  /-
    ι : Type u_1
    inst✝⁷ : Preorder ι
    F₁ : ι → Type u_2
    F₂ : ι → Type u_3
    F : ι → Type u_4
    X : ι → Type u_5
    T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
    f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
    inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
    inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
    T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
    f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
    inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
    inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
    T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
    f : (i j : ι) → (h : LE.le i j) → T h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
    inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : Sort u_9
    ih : (i : ι) → F₁ i → F₂ i → C
    compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
    z : Sigma fun i => F₁ i
    w : Sigma fun i => F₂ i
    j : ι
    hzj : LE.le z.fst j
    hwj : LE.le w.fst j
    k : ι
    hzk : LE.le z.fst k
    hwk : LE.le w.fst k
    i : ι
    hji : LE.le j i
    hki : LE.le k i
    ⊢ Eq (ih j ((f₁ z.fst j hzj) z.snd) ((f₂ w.fst j hwj) w.snd)) (ih k ((f₁ z.fst …
  -/
  simp_rw [compat _ _ hji, compat _ _ hki, map_map']
  /-
    🎉 no goals
  -/


/-- To define a binary function from the direct limit, it suffices to provide one binary function
from each component subject to a compatibility condition. -/
protected noncomputable def lift₂ (z : DirectLimit F₁ f₁) (w : DirectLimit F₂ f₂) : C :=
  z.hrecOn₂ w (φ := fun _ _ ↦ C) (lift₂Aux f₁ f₂ ih compat · ·)
    fun _ _ _ _ ⟨j, hx, hyj, jeq⟩ ⟨k, hyk, hz, keq⟩ ↦ heq_of_eq <| by
      /-
        ι : Type u_1
        inst✝⁷ : Preorder ι
        F₁ : ι → Type u_2
        F₂ : ι → Type u_3
        F : ι → Type u_4
        X : ι → Type u_5
        T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
        f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
        inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
        inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
        T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
        f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
        inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
        inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
        T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
        f : (i j : ι) → (h : LE.le i j) → T h
        inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
        inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
        inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
        C : Sort u_9
        ih : (i : ι) → F₁ i → F₂ i → C
        compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
        z : DirectLimit F₁ f₁
        w : DirectLimit F₂ f₂
        x✝⁵ : Sigma fun i => F₁ i
        x✝⁴ : Sigma fun i => F₂ i
        x✝³ : Sigma fun i => F₁ i
        x✝² : Sigma fun i => F₂ i
        x✝¹ : HasEquiv.Equiv x✝⁵ x✝³
        x✝ : HasEquiv.Equiv x✝⁴ x✝²
        j : ι
        hx : LE.le x✝⁵.fst j
        hyj : LE.le x✝³.fst j
        jeq : Eq ((f₁ x✝⁵.fst j hx) x✝⁵.snd) ((f₁ x✝³.fst j hyj) x✝³.snd)
        k : ι
        hyk : LE.le x✝⁴.fst k
        hz : LE.le x✝².fst k
        keq : Eq ((f₂ x✝⁴.fst k hyk) x✝⁴.snd) ((f₂ x✝².fst k hz) x✝².snd)
        ⊢ Eq ((fun x1 x2 => ↑(DirectLimit.lift₂Aux f₁ f₂ ih compat x1 x2)) x✝⁵ x✝⁴) (( …
      -/
      have ⟨i, hji, hki⟩ := exists_ge_ge j k
      simp_rw [(lift₂Aux ..).2 _ (hx.trans hji) (hyk.trans hki),
        (lift₂Aux ..).2 _ (hyj.trans hji) (hz.trans hki),
        ← map_map' _ hx hji, jeq, ← map_map' _ hz hki, ← keq, map_map']


theorem lift₂_def₂ (x : Σ i, F₁ i) (y : Σ i, F₂ i) (i) (hxi : x.1 ≤ i) (hyi : y.1 ≤ i) :
    DirectLimit.lift₂ f₁ f₂ ih compat ⟦x⟧ ⟦y⟧ = ih i (f₁ _ _ hxi x.2) (f₂ _ _ hyi y.2) :=
  (lift₂Aux _ _ _ compat _ _).2 ..


theorem lift₂_def (i x y) : DirectLimit.lift₂ f₁ f₂ ih compat ⟦⟨i, x⟩⟧ ⟦⟨i, y⟩⟧ = ih i x y := by
  /-
    ι : Type u_1
    inst✝⁵ : Preorder ι
    F₁ : ι → Type u_2
    F₂ : ι → Type u_3
    T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
    f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
    inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
    inst✝³ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
    T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
    f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
    inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
    inst✝¹ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    C : Sort u_9
    ih : (i : ι) → F₁ i → F₂ i → C
    compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq (ih i x y) (ih  …
    i : ι
    x : F₁ i
    y : F₂ i
    ⊢ Eq (DirectLimit.lift₂ f₁ f₂ ih compat (Quotient.mk (DirectLimit.setoid f₁) ⟨ …
  -/
  rw [lift₂_def₂ _ _ _ _ _ _ i le_rfl le_rfl, map_self', map_self']
  /-
    🎉 no goals
  -/


/-- To define a function from the direct limit, it suffices to provide one function from each
component subject to a compatibility condition. -/
noncomputable def map₂ : DirectLimit F₁ f₁ → DirectLimit F₂ f₂ → DirectLimit F f :=
  DirectLimit.lift₂ f₁ f₂ (fun i x y ↦ ⟦⟨i, ih i x y⟩⟧) fun j k h x y ↦ Quotient.sound <|
    have ⟨i, hji, hki⟩ := exists_ge_ge j k
                     /-
                       ι : Type u_1
                       inst✝⁷ : Preorder ι
                       F₁ : ι → Type u_2
                       F₂ : ι → Type u_3
                       F : ι → Type u_4
                       X : ι → Type u_5
                       T₁ : ⦃i j : ι⦄ → LE.le i j → Sort u_6
                       f₁ : (i j : ι) → (h : LE.le i j) → T₁ h
                       inst✝⁶ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₁ h) (F₁ i) (F₁ j)
                       inst✝⁵ : DirectedSystem F₁ fun x1 x2 x3 => ⇑(f₁ x1 x2 x3)
                       T₂ : ⦃i j : ι⦄ → LE.le i j → Sort u_7
                       f₂ : (i j : ι) → (h : LE.le i j) → T₂ h
                       inst✝⁴ : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T₂ h) (F₂ i) (F₂ j)
                       inst✝³ : DirectedSystem F₂ fun x1 x2 x3 => ⇑(f₂ x1 x2 x3)
                       T : ⦃i j : ι⦄ → LE.le i j → Sort u_8
                       f : (i j : ι) → (h : LE.le i j) → T h
                       inst✝² : ⦃i j : ι⦄ → (h : LE.le i j) → FunLike (T h) (F i) (F j)
                       inst✝¹ : DirectedSystem F fun x1 x2 x3 => ⇑(f x1 x2 x3)
                       inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                       ih : (i : ι) → F₁ i → F₂ i → F i
                       compat : ∀ (i j : ι) (h : LE.le i j) (x : F₁ i) (y : F₂ i), Eq ((f i j h) (ih  …
                       j k : ι
                       h : LE.le j k
                       x : F₁ j
                       y : F₂ j
                       i : ι
                       hji : LE.le j i
                       hki : LE.le k i
                       ⊢ Eq ((f ⟨j, ih j x y⟩.fst i hji) ⟨j, ih j x y⟩.snd) ((f ⟨k, ih k ((f₁ j k h)  …
                     -/
    ⟨i, hji, hki, by simp_rw [compat, map_map']⟩
                     /-
                       🎉 no goals
                     -/


theorem map₂_def₂ (x y) (i) (hxi : x.1 ≤ i) (hyi : y.1 ≤ i) :
    map₂ f₁ f₂ f ih compat ⟦x⟧ ⟦y⟧ = ⟦⟨i, ih i (f₁ _ _ hxi x.2) (f₂ _ _ hyi y.2)⟩⟧ :=
  lift₂_def₂ ..


theorem map₂_def (i x y) : map₂ f₁ f₂ f ih compat ⟦⟨i, x⟩⟧ ⟦⟨i, y⟩⟧ = ⟦⟨i, ih i x y⟩⟧ :=
  lift₂_def ..


/-- A inverse system indexed by a preorder is a contravariant functor from the preorder
to another category. It is dual to `DirectedSystem`. -/
class InverseSystem : Prop where
  map_self ⦃i⦄ (x : F i) : f le_rfl x = x
  map_map ⦃k j i⦄ (hkj : k ≤ j) (hji : j ≤ i) (x : F i) : f hkj (f hji x) = f (hkj.trans hji) x


/-- The inverse limit of an inverse system of types. -/
def limit (i : ι) : Set (∀ l : Iio i, F l) :=
  {F | ∀ ⦃j k⦄ (h : j.1 ≤ k.1), f h (F k) = F j}


/-- For a family of types `X` indexed by an preorder `ι` and an element `i : ι`,
`piLT X i` is the product of all the types indexed by elements below `i`. -/
abbrev piLT (X : ι → Type*) (i : ι) := ∀ l : Iio i, X l


/-- The projection from a Pi type to the Pi type over an initial segment of its indexing type. -/
abbrev piLTProj (f : piLT X j) : piLT X i := fun l ↦ f ⟨l, l.2.trans_le h⟩


theorem piLTProj_intro {l : Iio j} {f : piLT X j} (hl : l < i) :
    f l = piLTProj h f ⟨l, hl⟩ := rfl


/-- The predicate that says a family of equivalences between `F j` and `piLT X j`
  is a natural transformation. -/
def IsNatEquiv {s : Set ι} (equiv : ∀ j : s, F j ≃ piLT X j) : Prop :=
  ∀ ⦃j k⦄ (hj : j ∈ s) (hk : k ∈ s) (h : k ≤ j) (x : F j),
    equiv ⟨k, hk⟩ (f h x) = piLTProj h (equiv ⟨j, hj⟩ x)


/-- If `i` is a limit in a well-ordered type indexing a family of types,
then `piLT X i` is the limit of all `piLT X j` for `j < i`. -/
@[simps apply] noncomputable def piLTLim : piLT X i ≃ limit (piLTProj (X := X)) i where
  toFun f := ⟨fun j ↦ piLTProj j.2.le f, fun _ _ _ ↦ rfl⟩
  invFun f l := let k := hi.mid l.2; f.1 ⟨k, k.2.2⟩ ⟨l, k.2.1⟩
  left_inv f := rfl
  right_inv f := by
    /-
      ι✝ : Type u_1
      inst✝¹ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F j → F i
      i✝ j : ι✝
      h : LE.le i✝ j
      ι : Type u_6
      inst✝ : LinearOrder ι
      X : ι → Type u_7
      i : ι
      hi : Order.IsSuccPrelimit i
      f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
      ⊢ Eq
          ((fun f => ⟨fun j => InverseSystem.piLTProj ⋯ f, ⋯⟩)
            ((fun f l =>
                let k := hi.mid ⋯;
                ↑f ⟨↑k, ⋯⟩ ⟨↑l, ⋯⟩)
              f))
          f
    -/
    ext j l
    /-
      case a.h.h
      ι✝ : Type u_1
      inst✝¹ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F j → F i
      i✝ j✝ : ι✝
      h : LE.le i✝ j✝
      ι : Type u_6
      inst✝ : LinearOrder ι
      X : ι → Type u_7
      i : ι
      hi : Order.IsSuccPrelimit i
      f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
      j : ↑(Set.Iio i)
      l : ↑(Set.Iio ↑j)
      ⊢ Eq
          (↑((fun f => ⟨fun j => InverseSystem.piLTProj ⋯ f, ⋯⟩)
                ((fun f l =>
                    let k := hi.mid ⋯;
                    ↑f ⟨↑k, ⋯⟩ ⟨↑l, ⋯⟩)
                  f))
            j l)
          (↑f j l)
    -/
    set k := hi.mid (l.2.trans j.2)
    /-
      case a.h.h
      ι✝ : Type u_1
      inst✝¹ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F j → F i
      i✝ j✝ : ι✝
      h : LE.le i✝ j✝
      ι : Type u_6
      inst✝ : LinearOrder ι
      X : ι → Type u_7
      i : ι
      hi : Order.IsSuccPrelimit i
      f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
      j : ↑(Set.Iio i)
      l : ↑(Set.Iio ↑j)
      k : ↑(Set.Ioo (↑l) i) := hi.mid ⋯
      ⊢ Eq
          (↑((fun f => ⟨fun j => InverseSystem.piLTProj ⋯ f, ⋯⟩)
                ((fun f l =>
                    let k := hi.mid ⋯;
                    ↑f ⟨↑k, ⋯⟩ ⟨↑l, ⋯⟩)
                  f))
            j l)
          (↑f j l)
    -/
    obtain le | le := le_total j ⟨k, k.2.2⟩
    /-
      case a.h.h.inl
      ι✝ : Type u_1
      inst✝¹ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F j → F i
      i✝ j✝ : ι✝
      h : LE.le i✝ j✝
      ι : Type u_6
      inst✝ : LinearOrder ι
      X : ι → Type u_7
      i : ι
      hi : Order.IsSuccPrelimit i
      f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
      j : ↑(Set.Iio i)
      l : ↑(Set.Iio ↑j)
      k : ↑(Set.Ioo (↑l) i) := hi.mid ⋯
      le : LE.le j ⟨↑k, ⋯⟩
      ⊢ Eq
          (↑((fun f => ⟨fun j => InverseSystem.piLTProj ⋯ f, ⋯⟩)
                ((fun f l =>
                    let k := hi.mid ⋯;
                    ↑f ⟨↑k, ⋯⟩ ⟨↑l, ⋯⟩)
                  f))
            j l)
          (↑f j l)
    -/
    exacts [congr_fun (f.2 le) l, (congr_fun (f.2 le) ⟨l, _⟩).symm]
    /-
      🎉 no goals
    -/


theorem piLTLim_symm_apply {f} (k : Iio i) {l : Iio i} (hl : l.1 < k.1) :
    (piLTLim (X := X) hi).symm f l = f.1 k ⟨l, hl⟩ := by
  /-
    ι : Type u_6
    inst✝ : LinearOrder ι
    X : ι → Type u_7
    i : ι
    hi : Order.IsSuccPrelimit i
    f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
    k l : ↑(Set.Iio i)
    hl : LT.lt ↑l ↑k
    ⊢ Eq ((InverseSystem.piLTLim hi).symm f l) (↑f k ⟨↑l, hl⟩)
  -/
  conv_rhs => rw [← (piLTLim hi).right_inv f]
  /-
    ι : Type u_6
    inst✝ : LinearOrder ι
    X : ι → Type u_7
    i : ι
    hi : Order.IsSuccPrelimit i
    f : ↑(InverseSystem.limit InverseSystem.piLTProj i)
    k l : ↑(Set.Iio i)
    hl : LT.lt ↑l ↑k
    ⊢ Eq ((InverseSystem.piLTLim hi).symm f l) (↑((InverseSystem.piLTLim hi).toFun …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Splitting off the `X i` factor from the Pi type over `{j | j ≤ i}`. -/
def piSplitLE : piLT X i × X i ≃ ∀ j : Iic i, X j where
  toFun f j := if h : j = i then h.symm ▸ f.2 else f.1 ⟨j, j.2.lt_of_ne h⟩
  invFun f := (fun j ↦ f ⟨j, j.2.le⟩, f ⟨i, le_rfl⟩)
                   /-
                     ι✝ : Type u_1
                     inst✝² : Preorder ι✝
                     F₁ : ι✝ → Type u_2
                     F₂ : ι✝ → Type u_3
                     F✝ : ι✝ → Type u_4
                     X✝ : ι✝ → Type u_5
                     f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                     i✝ j : ι✝
                     h : LE.le i✝ j
                     ι : Type u_6
                     F : ι → Type u_7
                     X : ι → Type u_8
                     i : ι
                     inst✝¹ : PartialOrder ι
                     inst✝ : DecidableEq ι
                     f : Prod (InverseSystem.piLT X i) (X i)
                     ⊢ Eq ((fun f => { fst := fun j => f ⟨↑j, ⋯⟩, snd := f ⟨i, ⋯⟩ }) ((fun f j => d …
                   -/
  left_inv f := by ext j; exacts [dif_neg j.2.ne, dif_pos rfl]
                          /-
                            🎉 no goals
                          -/
  right_inv f := by
    /-
      ι✝ : Type u_1
      inst✝² : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F✝ : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
      i✝ j : ι✝
      h : LE.le i✝ j
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝¹ : PartialOrder ι
      inst✝ : DecidableEq ι
      f : (j : ↑(Set.Iic i)) → X ↑j
      ⊢ Eq ((fun f j => dite (Eq (↑j) i) (fun h => Eq.rec f.2 ⋯) fun h => f.1 ⟨↑j, ⋯ …
    -/
    ext j; dsimp only; split_ifs with h
      /-
        case pos
        ι✝ : Type u_1
        inst✝² : Preorder ι✝
        F₁ : ι✝ → Type u_2
        F₂ : ι✝ → Type u_3
        F✝ : ι✝ → Type u_4
        X✝ : ι✝ → Type u_5
        f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
        i✝ j✝ : ι✝
        h✝ : LE.le i✝ j✝
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝¹ : PartialOrder ι
        inst✝ : DecidableEq ι
        f : (j : ↑(Set.Iic i)) → X ↑j
        j : ↑(Set.Iic i)
        h : Eq (↑j) i
        ⊢ Eq (Eq.rec (f ⟨i, ⋯⟩) ⋯) (f j)
      -/
    · cases (Subtype.ext h : j = ⟨i, le_rfl⟩); rfl
                                               /-
                                                 🎉 no goals
                                               -/
      /-
        case neg
        ι✝ : Type u_1
        inst✝² : Preorder ι✝
        F₁ : ι✝ → Type u_2
        F₂ : ι✝ → Type u_3
        F✝ : ι✝ → Type u_4
        X✝ : ι✝ → Type u_5
        f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
        i✝ j✝ : ι✝
        h✝ : LE.le i✝ j✝
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝¹ : PartialOrder ι
        inst✝ : DecidableEq ι
        f : (j : ↑(Set.Iic i)) → X ↑j
        j : ↑(Set.Iic i)
        h : Not (Eq (↑j) i)
        ⊢ Eq (f ⟨↑j, ⋯⟩) (f j)
      -/
    · rfl
      /-
        🎉 no goals
      -/


@[simp] theorem piSplitLE_eq {f : piLT X i × X i} :
                                        /-
                                          ι : Type u_6
                                          X : ι → Type u_8
                                          i : ι
                                          inst✝¹ : PartialOrder ι
                                          inst✝ : DecidableEq ι
                                          f : Prod (InverseSystem.piLT X i) (X i)
                                          ⊢ Eq (InverseSystem.piSplitLE f ⟨i, ⋯⟩) f.2
                                        -/
    piSplitLE f ⟨i, le_rfl⟩ = f.2 := by simp [piSplitLE]
                                        /-
                                          🎉 no goals
                                        -/


theorem piSplitLE_lt {f : piLT X i × X i} {j} (hj : j < i) :
    piSplitLE f ⟨j, hj.le⟩ = f.1 ⟨j, hj⟩ := dif_neg hj.ne


local postfix:max "⁺" => succ -- Note: conflicts with `PosPart` notation


/-- Extend a family of bijections to `piLT` by one step. -/
def piEquivSucc : ∀ j : Iic i⁺, F j ≃ piLT X j :=
  piSplitLE (X := fun i ↦ F i ≃ piLT X i)
  (fun j ↦ equiv ⟨j, (lt_succ_iff_of_not_isMax hi).mp j.2⟩,
    e.trans <| ((equiv ⟨i, le_rfl⟩).prodCongr <| Equiv.refl _).trans <| piSplitLE.trans <|
      Equiv.piCongrSet <| Set.ext fun _ ↦ (lt_succ_iff_of_not_isMax hi).symm)


theorem piEquivSucc_self {x} :
    piEquivSucc equiv e hi ⟨_, le_rfl⟩ x ⟨i, lt_succ_of_not_isMax hi⟩ = (e x).2 := by
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    i : ι
    inst✝¹ : LinearOrder ι
    inst✝ : SuccOrder ι
    equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
    e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
    hi : Not (IsMax i)
    x : F ↑⟨Order.succ i, ⋯⟩
    ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨Order.succ i, ⋯⟩) x ⟨i, ⋯⟩) (e x).2
  -/
  simp [piEquivSucc]
  /-
    🎉 no goals
  -/


theorem isNatEquiv_piEquivSucc [InverseSystem f] (H : ∀ x, (e x).1 = f (le_succ i) x)
    (nat : IsNatEquiv f equiv) : IsNatEquiv f (piEquivSucc equiv e hi) := fun j k hj hk h x ↦ by
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    i : ι
    inst✝² : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝¹ : SuccOrder ι
    equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
    e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
    hi : Not (IsMax i)
    inst✝ : InverseSystem f
    H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
    nat : InverseSystem.IsNatEquiv f equiv
    j k : ι
    hj : Membership.mem (Set.Iic (Order.succ i)) j
    hk : Membership.mem (Set.Iic (Order.succ i)) k
    h : LE.le k j
    x : F j
    ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨k, hk⟩) (f h x)) (InverseSystem.p …
  -/
  have lt_succ {j} := (lt_succ_iff_of_not_isMax (b := j) hi).mpr
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    i : ι
    inst✝² : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝¹ : SuccOrder ι
    equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
    e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
    hi : Not (IsMax i)
    inst✝ : InverseSystem f
    H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
    nat : InverseSystem.IsNatEquiv f equiv
    j k : ι
    hj : Membership.mem (Set.Iic (Order.succ i)) j
    hk : Membership.mem (Set.Iic (Order.succ i)) k
    h : LE.le k j
    x : F j
    lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
    ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨k, hk⟩) (f h x)) (InverseSystem.p …
  -/
  obtain rfl | hj := le_succ_iff_eq_or_le.mp hj
    /-
      case inl
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
      e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
      hi : Not (IsMax i)
      inst✝ : InverseSystem f
      H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
      nat : InverseSystem.IsNatEquiv f equiv
      k : ι
      hk : Membership.mem (Set.Iic (Order.succ i)) k
      lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
      hj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
      h : LE.le k (Order.succ i)
      x : F (Order.succ i)
      ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨k, hk⟩) (f h x)) (InverseSystem.p …
    -/
  · obtain rfl | hk := le_succ_iff_eq_or_le.mp hk
      /-
        case inl.inl
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝² : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝¹ : SuccOrder ι
        equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
        e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
        hi : Not (IsMax i)
        inst✝ : InverseSystem f
        H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
        nat : InverseSystem.IsNatEquiv f equiv
        lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
        hj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
        x : F (Order.succ i)
        hk : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
        h : LE.le (Order.succ i) (Order.succ i)
        ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨Order.succ i, hk⟩) (f h x)) (Inve …
      -/
    · simp [InverseSystem.map_self]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝² : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝¹ : SuccOrder ι
        equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
        e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
        hi : Not (IsMax i)
        inst✝ : InverseSystem f
        H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
        nat : InverseSystem.IsNatEquiv f equiv
        k : ι
        hk✝ : Membership.mem (Set.Iic (Order.succ i)) k
        lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
        hj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
        h : LE.le k (Order.succ i)
        x : F (Order.succ i)
        hk : LE.le k i
        ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨k, hk✝⟩) (f h x)) (InverseSystem. …
      -/
    · funext l
      rw [piEquivSucc, piSplitLE_lt (lt_succ hk),
        ← InverseSystem.map_map (f := f) hk (le_succ i), ← H, piLTProj, nat le_rfl]
      /-
        case inl.inr.h
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝² : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝¹ : SuccOrder ι
        equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
        e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
        hi : Not (IsMax i)
        inst✝ : InverseSystem f
        H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
        nat : InverseSystem.IsNatEquiv f equiv
        k : ι
        hk✝ : Membership.mem (Set.Iic (Order.succ i)) k
        lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
        hj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
        h : LE.le k (Order.succ i)
        x : F (Order.succ i)
        hk : LE.le k i
        l : ↑(Set.Iio ↑⟨k, hk✝⟩)
        ⊢ Eq (InverseSystem.piLTProj hk ((equiv ⟨i, ⋯⟩) (e x).1) l) ((InverseSystem.pi …
      -/
      simp [piSplitLE_lt (l.2.trans_le hk)]
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equiv : (j : ↑(Set.Iic i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
      e : Equiv (F (Order.succ i)) (Prod (F i) (X i))
      hi : Not (IsMax i)
      inst✝ : InverseSystem f
      H : ∀ (x : F (Order.succ i)), Eq (e x).1 (f ⋯ x)
      nat : InverseSystem.IsNatEquiv f equiv
      j k : ι
      hj✝ : Membership.mem (Set.Iic (Order.succ i)) j
      hk : Membership.mem (Set.Iic (Order.succ i)) k
      h : LE.le k j
      x : F j
      lt_succ : ∀ {j : ι}, LE.le j i → LT.lt j (Order.succ i)
      hj : LE.le j i
      ⊢ Eq ((InverseSystem.piEquivSucc equiv e hi ⟨k, hk⟩) (f h x)) (InverseSystem.p …
    -/
  · rw [piEquivSucc, piSplitLE_lt (h.trans_lt <| lt_succ hj), nat hj, piSplitLE_lt (lt_succ hj)]
    /-
      🎉 no goals
    -/


/-- A natural family of bijections below a limit ordinal
induces a bijection at the limit ordinal. -/
@[simps] def invLimEquiv : limit f i ≃ limit (piLTProj (X := X)) i where
                                                                 /-
                                                                   ι✝ : Type u_1
                                                                   inst✝¹ : Preorder ι✝
                                                                   F₁ : ι✝ → Type u_2
                                                                   F₂ : ι✝ → Type u_3
                                                                   F✝ : ι✝ → Type u_4
                                                                   X✝ : ι✝ → Type u_5
                                                                   f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                                                                   i✝ j : ι✝
                                                                   h✝ : LE.le i✝ j
                                                                   ι : Type u_6
                                                                   F : ι → Type u_7
                                                                   X : ι → Type u_8
                                                                   i : ι
                                                                   inst✝ : LinearOrder ι
                                                                   f : ⦃i j : ι⦄ → LE.le i j → F j → F i
                                                                   equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
                                                                   nat : InverseSystem.IsNatEquiv f equiv
                                                                   t : ↑(InverseSystem.limit f i)
                                                                   x✝¹ x✝ : ↑(Set.Iio i)
                                                                   h : LE.le ↑x✝¹ ↑x✝
                                                                   ⊢ Eq ((fun l => (equiv l) (↑t l)) x✝¹) (InverseSystem.piLTProj h ((fun l => (e …
                                                                 -/
  toFun t := ⟨fun l ↦ equiv l (t.1 l), fun _ _ h ↦ Eq.symm <| by simp_rw [← t.2 h]; apply nat⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  invFun t := ⟨fun l ↦ (equiv l).symm (t.1 l),
                                                  /-
                                                    ι✝ : Type u_1
                                                    inst✝¹ : Preorder ι✝
                                                    F₁ : ι✝ → Type u_2
                                                    F₂ : ι✝ → Type u_3
                                                    F✝ : ι✝ → Type u_4
                                                    X✝ : ι✝ → Type u_5
                                                    f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                                                    i✝ j : ι✝
                                                    h✝ : LE.le i✝ j
                                                    ι : Type u_6
                                                    F : ι → Type u_7
                                                    X : ι → Type u_8
                                                    i : ι
                                                    inst✝ : LinearOrder ι
                                                    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
                                                    equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
                                                    nat : InverseSystem.IsNatEquiv f equiv
                                                    t : ↑(InverseSystem.limit InverseSystem.piLTProj i)
                                                    x✝¹ x✝ : ↑(Set.Iio i)
                                                    h : LE.le ↑x✝¹ ↑x✝
                                                    ⊢ Eq ((equiv x✝¹) (f h ((fun l => (equiv l).symm (↑t l)) x✝))) (↑t x✝¹)
                                                  -/
    fun _ _ h ↦ (Equiv.eq_symm_apply _).mpr <| by rw [nat, ← t.2 h]; simp⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                   /-
                     ι✝ : Type u_1
                     inst✝¹ : Preorder ι✝
                     F₁ : ι✝ → Type u_2
                     F₂ : ι✝ → Type u_3
                     F✝ : ι✝ → Type u_4
                     X✝ : ι✝ → Type u_5
                     f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                     i✝ j : ι✝
                     h : LE.le i✝ j
                     ι : Type u_6
                     F : ι → Type u_7
                     X : ι → Type u_8
                     i : ι
                     inst✝ : LinearOrder ι
                     f : ⦃i j : ι⦄ → LE.le i j → F j → F i
                     equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
                     nat : InverseSystem.IsNatEquiv f equiv
                     t : ↑(InverseSystem.limit f i)
                     ⊢ Eq ((fun t => ⟨fun l => (equiv l).symm (↑t l), ⋯⟩) ((fun t => ⟨fun l => (equ …
                   -/
  left_inv t := by ext; apply Equiv.left_inv
                        /-
                          🎉 no goals
                        -/
                    /-
                      ι✝ : Type u_1
                      inst✝¹ : Preorder ι✝
                      F₁ : ι✝ → Type u_2
                      F₂ : ι✝ → Type u_3
                      F✝ : ι✝ → Type u_4
                      X✝ : ι✝ → Type u_5
                      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                      i✝ j : ι✝
                      h : LE.le i✝ j
                      ι : Type u_6
                      F : ι → Type u_7
                      X : ι → Type u_8
                      i : ι
                      inst✝ : LinearOrder ι
                      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
                      equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
                      nat : InverseSystem.IsNatEquiv f equiv
                      t : ↑(InverseSystem.limit InverseSystem.piLTProj i)
                      ⊢ Eq ((fun t => ⟨fun l => (equiv l) (↑t l), ⋯⟩) ((fun t => ⟨fun l => (equiv l) …
                    -/
  right_inv t := by ext1; ext1; apply Equiv.right_inv
                                /-
                                  🎉 no goals
                                -/


/-- Extend a natural family of bijections to a limit ordinal. -/
noncomputable def piEquivLim : ∀ j : Iic i, F j ≃ piLT X j :=
  piSplitLE (X := fun j ↦ F j ≃ piLT X j)
    (equiv, equivLim.trans <| (invLimEquiv nat).trans (piLTLim hi).symm)


theorem isNatEquiv_piEquivLim [InverseSystem f] (H : ∀ x l, (equivLim x).1 l = f l.2.le x) :
    IsNatEquiv f (piEquivLim nat equivLim hi) := fun j k hj hk h t ↦ by
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    i : ι
    inst✝¹ : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
    nat : InverseSystem.IsNatEquiv f equiv
    equivLim : Equiv (F i) ↑(InverseSystem.limit f i)
    hi : Order.IsSuccPrelimit i
    inst✝ : InverseSystem f
    H : ∀ (x : F i) (l : ↑(Set.Iio i)), Eq (↑(equivLim x) l) (f ⋯ x)
    j k : ι
    hj : Membership.mem (Set.Iic i) j
    hk : Membership.mem (Set.Iic i) k
    h : LE.le k j
    t : F j
    ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk⟩) (f h t)) (InverseSyst …
  -/
  obtain rfl | hj := hj.eq_or_lt
    /-
      case inl
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝¹ : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝ : InverseSystem f
      j k : ι
      h : LE.le k j
      t : F j
      equiv : (j_1 : ↑(Set.Iio j)) → Equiv (F ↑j_1) (InverseSystem.piLT X ↑j_1)
      nat : InverseSystem.IsNatEquiv f equiv
      equivLim : Equiv (F j) ↑(InverseSystem.limit f j)
      hi : Order.IsSuccPrelimit j
      H : ∀ (x : F j) (l : ↑(Set.Iio j)), Eq (↑(equivLim x) l) (f ⋯ x)
      hj : Membership.mem (Set.Iic j) j
      hk : Membership.mem (Set.Iic j) k
      ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk⟩) (f h t)) (InverseSyst …
    -/
  · obtain rfl | hk := hk.eq_or_lt
      /-
        case inl.inl
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        inst✝¹ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝ : InverseSystem f
        k : ι
        h : LE.le k k
        t : F k
        equiv : (j : ↑(Set.Iio k)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
        nat : InverseSystem.IsNatEquiv f equiv
        equivLim : Equiv (F k) ↑(InverseSystem.limit f k)
        hi : Order.IsSuccPrelimit k
        H : ∀ (x : F k) (l : ↑(Set.Iio k)), Eq (↑(equivLim x) l) (f ⋯ x)
        hj hk : Membership.mem (Set.Iic k) k
        ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk⟩) (f h t)) (InverseSyst …
      -/
    · simp [InverseSystem.map_self]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        inst✝¹ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝ : InverseSystem f
        j k : ι
        h : LE.le k j
        t : F j
        equiv : (j_1 : ↑(Set.Iio j)) → Equiv (F ↑j_1) (InverseSystem.piLT X ↑j_1)
        nat : InverseSystem.IsNatEquiv f equiv
        equivLim : Equiv (F j) ↑(InverseSystem.limit f j)
        hi : Order.IsSuccPrelimit j
        H : ∀ (x : F j) (l : ↑(Set.Iio j)), Eq (↑(equivLim x) l) (f ⋯ x)
        hj : Membership.mem (Set.Iic j) j
        hk✝ : Membership.mem (Set.Iic j) k
        hk : LT.lt k j
        ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk✝⟩) (f h t)) (InverseSys …
      -/
    · funext l
      /-
        case inl.inr.h
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        inst✝¹ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝ : InverseSystem f
        j k : ι
        h : LE.le k j
        t : F j
        equiv : (j_1 : ↑(Set.Iio j)) → Equiv (F ↑j_1) (InverseSystem.piLT X ↑j_1)
        nat : InverseSystem.IsNatEquiv f equiv
        equivLim : Equiv (F j) ↑(InverseSystem.limit f j)
        hi : Order.IsSuccPrelimit j
        H : ∀ (x : F j) (l : ↑(Set.Iio j)), Eq (↑(equivLim x) l) (f ⋯ x)
        hj : Membership.mem (Set.Iic j) j
        hk✝ : Membership.mem (Set.Iic j) k
        hk : LT.lt k j
        l : ↑(Set.Iio ↑⟨k, hk✝⟩)
        ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk✝⟩) (f h t) l) (InverseS …
      -/
      simp_rw [piEquivLim, piSplitLE_lt hk, piSplitLE_eq, Equiv.trans_apply]
      /-
        case inl.inr.h
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        inst✝¹ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝ : InverseSystem f
        j k : ι
        h : LE.le k j
        t : F j
        equiv : (j_1 : ↑(Set.Iio j)) → Equiv (F ↑j_1) (InverseSystem.piLT X ↑j_1)
        nat : InverseSystem.IsNatEquiv f equiv
        equivLim : Equiv (F j) ↑(InverseSystem.limit f j)
        hi : Order.IsSuccPrelimit j
        H : ∀ (x : F j) (l : ↑(Set.Iio j)), Eq (↑(equivLim x) l) (f ⋯ x)
        hj : Membership.mem (Set.Iic j) j
        hk✝ : Membership.mem (Set.Iic j) k
        hk : LT.lt k j
        l : ↑(Set.Iio ↑⟨k, hk✝⟩)
        ⊢ Eq ((equiv ⟨k, hk⟩) (f h t) l) (InverseSystem.piLTProj h ((InverseSystem.piL …
      -/
      rw [piLTProj, piLTLim_symm_apply hi ⟨k, hk⟩ (by exact l.2), invLimEquiv_apply_coe, H]
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝¹ : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      equiv : (j : ↑(Set.Iio i)) → Equiv (F ↑j) (InverseSystem.piLT X ↑j)
      nat : InverseSystem.IsNatEquiv f equiv
      equivLim : Equiv (F i) ↑(InverseSystem.limit f i)
      hi : Order.IsSuccPrelimit i
      inst✝ : InverseSystem f
      H : ∀ (x : F i) (l : ↑(Set.Iio i)), Eq (↑(equivLim x) l) (f ⋯ x)
      j k : ι
      hj✝ : Membership.mem (Set.Iic i) j
      hk : Membership.mem (Set.Iic i) k
      h : LE.le k j
      t : F j
      hj : LT.lt j i
      ⊢ Eq ((InverseSystem.piEquivLim nat equivLim hi ⟨k, hk⟩) (f h t)) (InverseSyst …
    -/
  · rw [piEquivLim, piSplitLE_lt (h.trans_lt hj), piSplitLE_lt hj]; apply nat
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- A natural partial family of bijections to `piLT` satisfying a compatibility condition. -/
@[ext] structure PEquivOn (s : Set ι) where
  /-- A partial family of bijections between `F` and `piLT X` defined on some set in `ι`. -/
  equiv (i : s) : F i ≃ piLT X i
  /-- It is a natural family of bijections. -/
  nat : IsNatEquiv f equiv
  /-- It is compatible with a family of bijections relating `F i⁺` to `F i`. -/
  compat {i} (hsi : (i⁺ : ι) ∈ s) (hi : ¬IsMax i) (x) :
    equiv ⟨i⁺, hsi⟩ x ⟨i, lt_succ_of_not_isMax hi⟩ = (equivSucc hi x).2


/-- Restrict a partial family of bijections to a smaller set. -/
@[simps] def PEquivOn.restrict (e : PEquivOn f equivSucc t) (h : s ⊆ t) :
    PEquivOn f equivSucc s where
  equiv i := e.equiv ⟨i, h i.2⟩
  nat _ _ _ _ := e.nat _ _
  compat _ := e.compat _


theorem unique_pEquivOn (hs : IsLowerSet s) {e₁ e₂ : PEquivOn f equivSucc s} : e₁ = e₂ := by
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    inst✝² : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝¹ : SuccOrder ι
    equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
    s : Set ι
    inst✝ : WellFoundedLT ι
    hs : IsLowerSet s
    e₁ e₂ : InverseSystem.PEquivOn f equivSucc s
    ⊢ Eq e₁ e₂
  -/
  obtain ⟨e₁, nat₁, compat₁⟩ := e₁
  /-
    case mk
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    inst✝² : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝¹ : SuccOrder ι
    equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
    s : Set ι
    inst✝ : WellFoundedLT ι
    hs : IsLowerSet s
    e₂ : InverseSystem.PEquivOn f equivSucc s
    e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
    nat₁ : InverseSystem.IsNatEquiv f e₁
    compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
    ⊢ Eq { equiv := e₁, nat := nat₁, compat := compat₁ } e₂
  -/
  obtain ⟨e₂, nat₂, compat₂⟩ := e₂
  /-
    case mk.mk
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    inst✝² : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝¹ : SuccOrder ι
    equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
    s : Set ι
    inst✝ : WellFoundedLT ι
    hs : IsLowerSet s
    e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
    nat₁ : InverseSystem.IsNatEquiv f e₁
    compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
    e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
    nat₂ : InverseSystem.IsNatEquiv f e₂
    compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
    ⊢ Eq { equiv := e₁, nat := nat₁, compat := compat₁ } { equiv := e₂, nat := nat …
  -/
  ext1; ext1 i; dsimp only
  refine SuccOrder.prelimitRecOn i.1 (C := fun i ↦ ∀ h : i ∈ s, e₁ ⟨i, h⟩ = e₂ ⟨i, h⟩)
    (fun i nmax ih hi ↦ ?_) (fun i lim ih hi ↦ ?_) i.2
    /-
      case mk.mk.equiv.h.refine_1
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      nmax : Not (IsMax i)
      ih : (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h⟩) (e₂ ⟨i, h⟩)) i
      hi : Membership.mem s (Order.succ i)
      ⊢ Eq (e₁ ⟨Order.succ i, hi⟩) (e₂ ⟨Order.succ i, hi⟩)
    -/
  · ext x ⟨j, hj⟩
    /-
      case mk.mk.equiv.h.refine_1.H.h.mk
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      nmax : Not (IsMax i)
      ih : (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h⟩) (e₂ ⟨i, h⟩)) i
      hi : Membership.mem s (Order.succ i)
      x : F ↑⟨Order.succ i, hi⟩
      j : ι
      hj : Membership.mem (Set.Iio ↑⟨Order.succ i, hi⟩) j
      ⊢ Eq ((e₁ ⟨Order.succ i, hi⟩) x ⟨j, hj⟩) ((e₂ ⟨Order.succ i, hi⟩) x ⟨j, hj⟩)
    -/
    obtain rfl | hj := ((lt_succ_iff_of_not_isMax nmax).mp hj).eq_or_lt
      /-
        case mk.mk.equiv.h.refine_1.H.h.mk.inl
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        inst✝² : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝¹ : SuccOrder ι
        equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
        s : Set ι
        inst✝ : WellFoundedLT ι
        hs : IsLowerSet s
        e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
        nat₁ : InverseSystem.IsNatEquiv f e₁
        compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
        e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
        nat₂ : InverseSystem.IsNatEquiv f e₂
        compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
        i : ↑s
        j : ι
        nmax : Not (IsMax j)
        ih : ∀ (h : Membership.mem s j), Eq (e₁ ⟨j, h⟩) (e₂ ⟨j, h⟩)
        hi : Membership.mem s (Order.succ j)
        x : F ↑⟨Order.succ j, hi⟩
        hj : Membership.mem (Set.Iio ↑⟨Order.succ j, hi⟩) j
        ⊢ Eq ((e₁ ⟨Order.succ j, hi⟩) x ⟨j, hj⟩) ((e₂ ⟨Order.succ j, hi⟩) x ⟨j, hj⟩)
      -/
    · exact (compat₁ _ nmax x).trans (compat₂ _ nmax x).symm
      /-
        🎉 no goals
      -/
    /-
      case mk.mk.equiv.h.refine_1.H.h.mk.inr
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      nmax : Not (IsMax i)
      ih : (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h⟩) (e₂ ⟨i, h⟩)) i
      hi : Membership.mem s (Order.succ i)
      x : F ↑⟨Order.succ i, hi⟩
      j : ι
      hj✝ : Membership.mem (Set.Iio ↑⟨Order.succ i, hi⟩) j
      hj : LT.lt j i
      ⊢ Eq ((e₁ ⟨Order.succ i, hi⟩) x ⟨j, hj✝⟩) ((e₂ ⟨Order.succ i, hi⟩) x ⟨j, hj✝⟩)
    -/
    have hi : i ∈ s := hs (le_succ i) hi
    rw [piLTProj_intro (f := e₁ _ x) (le_succ i) (by exact hj),
        ← nat₁ _ hi (by exact le_succ i), ih, nat₂ _ hi (by exact le_succ i)]
    /-
      case mk.mk.equiv.h.refine_2
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      lim : Order.IsSuccPrelimit i
      ih : ∀ (b : ι), LT.lt b i → (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h …
      hi : Membership.mem s i
      ⊢ Eq (e₁ ⟨i, hi⟩) (e₂ ⟨i, hi⟩)
    -/
  · ext x j
    /-
      case mk.mk.equiv.h.refine_2.H.h
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      lim : Order.IsSuccPrelimit i
      ih : ∀ (b : ι), LT.lt b i → (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h …
      hi : Membership.mem s i
      x : F ↑⟨i, hi⟩
      j : ↑(Set.Iio ↑⟨i, hi⟩)
      ⊢ Eq ((e₁ ⟨i, hi⟩) x j) ((e₂ ⟨i, hi⟩) x j)
    -/
    have ⟨k, hjk, hki⟩ := lim.mid j.2
    /-
      case mk.mk.equiv.h.refine_2.H.h
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      inst✝² : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝¹ : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s : Set ι
      inst✝ : WellFoundedLT ι
      hs : IsLowerSet s
      e₁ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₁ : InverseSystem.IsNatEquiv f e₁
      compat₁ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      e₂ : (i : ↑s) → Equiv (F ↑i) (InverseSystem.piLT X ↑i)
      nat₂ : InverseSystem.IsNatEquiv f e₂
      compat₂ : ∀ {i : ι} (hsi : Membership.mem s (Order.succ i)) (hi : Not (IsMax i …
      i✝ : ↑s
      i : ι
      lim : Order.IsSuccPrelimit i
      ih : ∀ (b : ι), LT.lt b i → (fun i => ∀ (h : Membership.mem s i), Eq (e₁ ⟨i, h …
      hi : Membership.mem s i
      x : F ↑⟨i, hi⟩
      j : ↑(Set.Iio ↑⟨i, hi⟩)
      k : ι
      hjk : LT.lt (↑j) k
      hki : LT.lt k i
      ⊢ Eq ((e₁ ⟨i, hi⟩) x j) ((e₂ ⟨i, hi⟩) x j)
    -/
    have hk : k ∈ s := hs hki.le hi
    rw [piLTProj_intro (f := e₁ _ x) hki.le hjk, piLTProj_intro (f := e₂ _ x) hki.le hjk,
      ← nat₁ _ hk, ← nat₂ _ hk, ih _ hki]


theorem pEquivOn_apply_eq (h : IsLowerSet (s ∩ t))
    {e₁ : PEquivOn f equivSucc s} {e₂ : PEquivOn f equivSucc t} {i} {his : i ∈ s} {hit : i ∈ t} :
    e₁.equiv ⟨i, his⟩ = e₂.equiv ⟨i, hit⟩ :=
  show (e₁.restrict inter_subset_left).equiv ⟨i, his, hit⟩ =
       (e₂.restrict inter_subset_right).equiv ⟨i, his, hit⟩ from
  congr_fun (congr_arg _ <| unique_pEquivOn h) _


/-- Extend a partial family of bijections by one step. -/
def pEquivOnSucc [InverseSystem f] (hi : ¬IsMax i) (e : PEquivOn f equivSucc (Iic i))
    (H : ∀ ⦃i⦄ (hi : ¬ IsMax i) x, (equivSucc hi x).1 = f (le_succ i) x) :
    PEquivOn f equivSucc (Iic i⁺) where
  equiv := piEquivSucc e.equiv (equivSucc hi) hi
  nat := isNatEquiv_piEquivSucc hi (H hi) e.nat
  compat hsj hj x := by
    /-
      ι✝ : Type u_1
      inst✝⁴ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F✝ : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
      i✝¹ j : ι✝
      h : LE.le i✝¹ j
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝³ : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝² : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s t : Set ι
      inst✝¹ : WellFoundedLT ι
      inst✝ : InverseSystem f
      hi : Not (IsMax i)
      e : InverseSystem.PEquivOn f equivSucc (Set.Iic i)
      H : ∀ ⦃i : ι⦄ (hi : Not (IsMax i)) (x : F (Order.succ i)), Eq ((equivSucc hi)  …
      i✝ : ι
      hsj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i✝)
      hj : Not (IsMax i✝)
      x : F ↑⟨Order.succ i✝, hsj⟩
      ⊢ Eq ((InverseSystem.piEquivSucc e.equiv (equivSucc hi) hi ⟨Order.succ i✝, hsj …
    -/
    obtain eq | lt := hsj.eq_or_lt
      /-
        case inl
        ι✝ : Type u_1
        inst✝⁴ : Preorder ι✝
        F₁ : ι✝ → Type u_2
        F₂ : ι✝ → Type u_3
        F✝ : ι✝ → Type u_4
        X✝ : ι✝ → Type u_5
        f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
        i✝¹ j : ι✝
        h : LE.le i✝¹ j
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝³ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝² : SuccOrder ι
        equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
        s t : Set ι
        inst✝¹ : WellFoundedLT ι
        inst✝ : InverseSystem f
        hi : Not (IsMax i)
        e : InverseSystem.PEquivOn f equivSucc (Set.Iic i)
        H : ∀ ⦃i : ι⦄ (hi : Not (IsMax i)) (x : F (Order.succ i)), Eq ((equivSucc hi)  …
        i✝ : ι
        hsj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i✝)
        hj : Not (IsMax i✝)
        x : F ↑⟨Order.succ i✝, hsj⟩
        eq : Eq (Order.succ i✝) (Order.succ i)
        ⊢ Eq ((InverseSystem.piEquivSucc e.equiv (equivSucc hi) hi ⟨Order.succ i✝, hsj …
      -/
    · cases (succ_eq_succ_iff_of_not_isMax hj hi).mp eq; simp [piEquivSucc]
                                                         /-
                                                           🎉 no goals
                                                         -/
      /-
        case inr
        ι✝ : Type u_1
        inst✝⁴ : Preorder ι✝
        F₁ : ι✝ → Type u_2
        F₂ : ι✝ → Type u_3
        F✝ : ι✝ → Type u_4
        X✝ : ι✝ → Type u_5
        f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
        i✝¹ j : ι✝
        h : LE.le i✝¹ j
        ι : Type u_6
        F : ι → Type u_7
        X : ι → Type u_8
        i : ι
        inst✝³ : LinearOrder ι
        f : ⦃i j : ι⦄ → LE.le i j → F j → F i
        inst✝² : SuccOrder ι
        equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
        s t : Set ι
        inst✝¹ : WellFoundedLT ι
        inst✝ : InverseSystem f
        hi : Not (IsMax i)
        e : InverseSystem.PEquivOn f equivSucc (Set.Iic i)
        H : ∀ ⦃i : ι⦄ (hi : Not (IsMax i)) (x : F (Order.succ i)), Eq ((equivSucc hi)  …
        i✝ : ι
        hsj : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i✝)
        hj : Not (IsMax i✝)
        x : F ↑⟨Order.succ i✝, hsj⟩
        lt : LT.lt (Order.succ i✝) (Order.succ i)
        ⊢ Eq ((InverseSystem.piEquivSucc e.equiv (equivSucc hi) hi ⟨Order.succ i✝, hsj …
      -/
    · rwa [piEquivSucc, piSplitLE_lt, e.compat]
      /-
        🎉 no goals
      -/


/-- Glue partial families of bijections at a limit ordinal,
obtaining a partial family over a right-open interval. -/
noncomputable def pEquivOnGlue : PEquivOn f equivSucc (Iio i) where
  equiv := (piLTLim (X := fun j ↦ F j ≃ piLT X j) hi).symm
    ⟨fun j ↦ ((e j).restrict fun _ h ↦ h.le).equiv, fun _ _ h ↦ funext fun _ ↦
      pEquivOn_apply_eq ((isLowerSet_Iio _).inter <| isLowerSet_Iio _)⟩
                        /-
                          ι✝ : Type u_1
                          inst✝³ : Preorder ι✝
                          F₁ : ι✝ → Type u_2
                          F₂ : ι✝ → Type u_3
                          F✝ : ι✝ → Type u_4
                          X✝ : ι✝ → Type u_5
                          f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
                          i✝ j✝ : ι✝
                          h✝ : LE.le i✝ j✝
                          ι : Type u_6
                          F : ι → Type u_7
                          X : ι → Type u_8
                          i : ι
                          inst✝² : LinearOrder ι
                          f : ⦃i j : ι⦄ → LE.le i j → F j → F i
                          inst✝¹ : SuccOrder ι
                          equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
                          s t : Set ι
                          inst✝ : WellFoundedLT ι
                          hi : Order.IsSuccPrelimit i
                          e : (j : ↑(Set.Iio i)) → InverseSystem.PEquivOn f equivSucc (Set.Iic ↑j)
                          j k : ι
                          hj : Membership.mem (Set.Iio i) j
                          hk : Membership.mem (Set.Iio i) k
                          h : LE.le k j
                          ⊢ ∀ (x : F j), Eq (((InverseSystem.piLTLim hi).symm ⟨fun j => ((e j).restrict  …
                        -/
  nat j k hj hk h := by rw [piLTLim_symm_apply]; exacts [(e _).nat _ _ _, h.trans_lt (hi.mid _).2.1]
                                                 /-
                                                   🎉 no goals
                                                 -/
  compat hj := have k := hi.mid hj
       /-
         ι✝ : Type u_1
         inst✝³ : Preorder ι✝
         F₁ : ι✝ → Type u_2
         F₂ : ι✝ → Type u_3
         F✝ : ι✝ → Type u_4
         X✝ : ι✝ → Type u_5
         f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
         i✝¹ j : ι✝
         h : LE.le i✝¹ j
         ι : Type u_6
         F : ι → Type u_7
         X : ι → Type u_8
         i : ι
         inst✝² : LinearOrder ι
         f : ⦃i j : ι⦄ → LE.le i j → F j → F i
         inst✝¹ : SuccOrder ι
         equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
         s t : Set ι
         inst✝ : WellFoundedLT ι
         hi : Order.IsSuccPrelimit i
         e : (j : ↑(Set.Iio i)) → InverseSystem.PEquivOn f equivSucc (Set.Iic ↑j)
         i✝ : ι
         hj : Membership.mem (Set.Iio i) (Order.succ i✝)
         k : ↑(Set.Ioo (Order.succ i✝) i)
         ⊢ ∀ (hi_1 : Not (IsMax i✝)) (x : F ↑⟨Order.succ i✝, hj⟩), Eq (((InverseSystem. …
       -/
    by rw [piLTLim_symm_apply hi ⟨_, k.2.2⟩ (by exact k.2.1)]; apply (e _).compat
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Extend `pEquivOnGlue` by one step, obtaining a partial family over a right-closed interval. -/
noncomputable def pEquivOnLim [InverseSystem f]
    (equivLim : F i ≃ limit f i) (H : ∀ x l, (equivLim x).1 l = f l.2.le x) :
    PEquivOn f equivSucc (Iic i) where
  equiv := piEquivLim (pEquivOnGlue hi e).nat equivLim hi
  nat := isNatEquiv_piEquivLim (pEquivOnGlue hi e).nat hi H
  compat hsj hj x := by
    /-
      ι✝ : Type u_1
      inst✝⁴ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F✝ : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
      i✝¹ j : ι✝
      h : LE.le i✝¹ j
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝³ : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝² : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s t : Set ι
      inst✝¹ : WellFoundedLT ι
      hi : Order.IsSuccPrelimit i
      e : (j : ↑(Set.Iio i)) → InverseSystem.PEquivOn f equivSucc (Set.Iic ↑j)
      inst✝ : InverseSystem f
      equivLim : Equiv (F i) ↑(InverseSystem.limit f i)
      H : ∀ (x : F i) (l : ↑(Set.Iio i)), Eq (↑(equivLim x) l) (f ⋯ x)
      i✝ : ι
      hsj : Membership.mem (Set.Iic i) (Order.succ i✝)
      hj : Not (IsMax i✝)
      x : F ↑⟨Order.succ i✝, hsj⟩
      ⊢ Eq ((InverseSystem.piEquivLim ⋯ equivLim hi ⟨Order.succ i✝, hsj⟩) x ⟨i✝, ⋯⟩) …
    -/
    rw [piEquivLim, piSplitLE_lt (hi.succ_lt <| (succ_le_iff_of_not_isMax hj).mp hsj)]
    /-
      ι✝ : Type u_1
      inst✝⁴ : Preorder ι✝
      F₁ : ι✝ → Type u_2
      F₂ : ι✝ → Type u_3
      F✝ : ι✝ → Type u_4
      X✝ : ι✝ → Type u_5
      f✝ : ⦃i j : ι✝⦄ → LE.le i j → F✝ j → F✝ i
      i✝¹ j : ι✝
      h : LE.le i✝¹ j
      ι : Type u_6
      F : ι → Type u_7
      X : ι → Type u_8
      i : ι
      inst✝³ : LinearOrder ι
      f : ⦃i j : ι⦄ → LE.le i j → F j → F i
      inst✝² : SuccOrder ι
      equivSucc : ⦃i : ι⦄ → Not (IsMax i) → Equiv (F (Order.succ i)) (Prod (F i) (X  …
      s t : Set ι
      inst✝¹ : WellFoundedLT ι
      hi : Order.IsSuccPrelimit i
      e : (j : ↑(Set.Iio i)) → InverseSystem.PEquivOn f equivSucc (Set.Iic ↑j)
      inst✝ : InverseSystem f
      equivLim : Equiv (F i) ↑(InverseSystem.limit f i)
      H : ∀ (x : F i) (l : ↑(Set.Iio i)), Eq (↑(equivLim x) l) (f ⋯ x)
      i✝ : ι
      hsj : Membership.mem (Set.Iic i) (Order.succ i✝)
      hj : Not (IsMax i✝)
      x : F ↑⟨Order.succ i✝, hsj⟩
      ⊢ Eq (({ fst := (InverseSystem.pEquivOnGlue hi e).equiv, snd := equivLim.trans …
    -/
    apply (pEquivOnGlue hi e).compat
    /-
      🎉 no goals
    -/


private noncomputable def globalEquivAux (i : ι) :
    PEquivOn f (fun i hi ↦ (equivSucc i hi).1) (Iic i) :=
  SuccOrder.prelimitRecOn i
    (fun _ hi e ↦ pEquivOnSucc hi e fun i hi ↦ (equivSucc i hi).2)
    fun i hi e ↦ pEquivOnLim hi (fun j ↦ e j j.2) (equivLim i hi).1 (equivLim i hi).2


/-- Over a well-ordered type, construct a family of bijections by transfinite recursion. -/
noncomputable def globalEquiv (i : ι) : F i ≃ piLT X i :=
  (globalEquivAux equivSucc equivLim i).equiv ⟨i, le_rfl⟩


theorem globalEquiv_naturality ⦃i j⦄ (h : i ≤ j) (x : F j) :
    letI e := globalEquiv equivSucc equivLim
    e i (f h x) = piLTProj h (e j x) := by
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    inst✝³ : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝² : WellFoundedLT ι
    inst✝¹ : SuccOrder ι
    inst✝ : InverseSystem f
    equivSucc : (i : ι) → Not (IsMax i) → Subtype fun e => ∀ (x : F (Order.succ i) …
    equivLim : (i : ι) → Order.IsSuccPrelimit i → Subtype fun e => ∀ (x : F i) (l  …
    i j : ι
    h : LE.le i j
    x : F j
    ⊢ Eq ((InverseSystem.globalEquiv equivSucc equivLim i) (f h x)) (InverseSystem …
  -/
  refine (DFunLike.congr_fun ?_ _).trans ((globalEquivAux equivSucc equivLim j).nat le_rfl h h x)
  /-
    ι : Type u_6
    F : ι → Type u_7
    X : ι → Type u_8
    inst✝³ : LinearOrder ι
    f : ⦃i j : ι⦄ → LE.le i j → F j → F i
    inst✝² : WellFoundedLT ι
    inst✝¹ : SuccOrder ι
    inst✝ : InverseSystem f
    equivSucc : (i : ι) → Not (IsMax i) → Subtype fun e => ∀ (x : F (Order.succ i) …
    equivLim : (i : ι) → Order.IsSuccPrelimit i → Subtype fun e => ∀ (x : F i) (l  …
    i j : ι
    h : LE.le i j
    x : F j
    ⊢ Eq (InverseSystem.globalEquiv equivSucc equivLim i) ((InverseSystem.globalEq …
  -/
  exact pEquivOn_apply_eq ((isLowerSet_Iic _).inter <| isLowerSet_Iic _)
  /-
    🎉 no goals
  -/


theorem globalEquiv_compatibility ⦃i⦄ (hi : ¬IsMax i) (x) :
    globalEquiv equivSucc equivLim i⁺ x ⟨i, lt_succ_of_not_isMax hi⟩ = ((equivSucc i hi).1 x).2 :=
  (globalEquivAux equivSucc equivLim i⁺).compat le_rfl hi x


