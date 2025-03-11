/-- The relative product homotopy of `homotopies` between functions `f` and `g` -/
@[simps!]
def HomotopyRel.pi (homotopies : ∀ i : I, HomotopyRel (f i) (g i) S) :
    HomotopyRel (pi f) (pi g) S :=
  { Homotopy.pi fun i => (homotopies i).toHomotopy with
    prop' := by
      /-
        I : Type u_1
        A : Type u_2
        X : I → Type u_3
        inst✝¹ : (i : I) → TopologicalSpace (X i)
        inst✝ : TopologicalSpace A
        f g : (i : I) → ContinuousMap A (X i)
        S : Set A
        homotopies : (i : I) → (f i).HomotopyRel (g i) S
        ⊢ ∀ (t : ↑unitInterval) (x : A), Membership.mem S x → Eq ({ toFun := fun x =>  …
      -/
      intro t x hx
      /-
        I : Type u_1
        A : Type u_2
        X : I → Type u_3
        inst✝¹ : (i : I) → TopologicalSpace (X i)
        inst✝ : TopologicalSpace A
        f g : (i : I) → ContinuousMap A (X i)
        S : Set A
        homotopies : (i : I) → (f i).HomotopyRel (g i) S
        t : ↑unitInterval
        x : A
        hx : Membership.mem S x
        ⊢ Eq ({ toFun := fun x => __src✝.toFun { fst := t, snd := x }, continuous_toFu …
      -/
      dsimp only [coe_mk, pi_eval, toFun_eq_coe, HomotopyWith.coe_toContinuousMap]
      /-
        I : Type u_1
        A : Type u_2
        X : I → Type u_3
        inst✝¹ : (i : I) → TopologicalSpace (X i)
        inst✝ : TopologicalSpace A
        f g : (i : I) → ContinuousMap A (X i)
        S : Set A
        homotopies : (i : I) → (f i).HomotopyRel (g i) S
        t : ↑unitInterval
        x : A
        hx : Membership.mem S x
        ⊢ Eq ((ContinuousMap.Homotopy.pi fun i => (homotopies i).toHomotopy).toContinu …
      -/
      simp only [funext_iff, ← forall_and]
      /-
        I : Type u_1
        A : Type u_2
        X : I → Type u_3
        inst✝¹ : (i : I) → TopologicalSpace (X i)
        inst✝ : TopologicalSpace A
        f g : (i : I) → ContinuousMap A (X i)
        S : Set A
        homotopies : (i : I) → (f i).HomotopyRel (g i) S
        t : ↑unitInterval
        x : A
        hx : Membership.mem S x
        ⊢ ∀ (x_1 : I), Eq ((ContinuousMap.Homotopy.pi fun i => (homotopies i).toHomoto …
      -/
      intro i
      /-
        I : Type u_1
        A : Type u_2
        X : I → Type u_3
        inst✝¹ : (i : I) → TopologicalSpace (X i)
        inst✝ : TopologicalSpace A
        f g : (i : I) → ContinuousMap A (X i)
        S : Set A
        homotopies : (i : I) → (f i).HomotopyRel (g i) S
        t : ↑unitInterval
        x : A
        hx : Membership.mem S x
        i : I
        ⊢ Eq ((ContinuousMap.Homotopy.pi fun i => (homotopies i).toHomotopy).toContinu …
      -/
      exact (homotopies i).prop' t x hx }
      /-
        🎉 no goals
      -/


/-- The product of homotopies `F` and `G`,
  where `F` takes `f₀` to `f₁` and `G` takes `g₀` to `g₁` -/
@[simps]
def Homotopy.prod (F : Homotopy f₀ f₁) (G : Homotopy g₀ g₁) :
    Homotopy (ContinuousMap.prodMk f₀ g₀) (ContinuousMap.prodMk f₁ g₁) where
  toFun t := (F t, G t)
                        /-
                          α : Type u_1
                          β : Type u_2
                          inst✝² : TopologicalSpace α
                          inst✝¹ : TopologicalSpace β
                          A : Type u_3
                          inst✝ : TopologicalSpace A
                          f₀ f₁ : ContinuousMap A α
                          g₀ g₁ : ContinuousMap A β
                          S : Set A
                          F : f₀.Homotopy f₁
                          G : g₀.Homotopy g₁
                          x : A
                          ⊢ Eq ({ toFun := fun t => { fst := F t, snd := G t }, continuous_toFun := ⋯ }. …
                        -/
  map_zero_left x := by simp only [prod_eval, Homotopy.apply_zero]
                        /-
                          🎉 no goals
                        -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝² : TopologicalSpace α
                         inst✝¹ : TopologicalSpace β
                         A : Type u_3
                         inst✝ : TopologicalSpace A
                         f₀ f₁ : ContinuousMap A α
                         g₀ g₁ : ContinuousMap A β
                         S : Set A
                         F : f₀.Homotopy f₁
                         G : g₀.Homotopy g₁
                         x : A
                         ⊢ Eq ({ toFun := fun t => { fst := F t, snd := G t }, continuous_toFun := ⋯ }. …
                       -/
  map_one_left x := by simp only [prod_eval, Homotopy.apply_one]
                       /-
                         🎉 no goals
                       -/


/-- The relative product of homotopies `F` and `G`,
  where `F` takes `f₀` to `f₁` and `G` takes `g₀` to `g₁` -/
@[simps!]
def HomotopyRel.prod (F : HomotopyRel f₀ f₁ S) (G : HomotopyRel g₀ g₁ S) :
    HomotopyRel (prodMk f₀ g₀) (prodMk f₁ g₁) S where
  toHomotopy := Homotopy.prod F.toHomotopy G.toHomotopy
  prop' t x hx := Prod.ext (F.prop' t x hx) (G.prop' t x hx)


local infixl:70 " ⬝ " => Quotient.comp


/-- The product of a family of path homotopies. This is just a specialization of `HomotopyRel`. -/
def piHomotopy (γ₀ γ₁ : ∀ i, Path (as i) (bs i)) (H : ∀ i, Path.Homotopy (γ₀ i) (γ₁ i)) :
    Path.Homotopy (Path.pi γ₀) (Path.pi γ₁) :=
  ContinuousMap.HomotopyRel.pi H


/-- The product of a family of path homotopy classes. -/
def pi (γ : ∀ i, Path.Homotopic.Quotient (as i) (bs i)) : Path.Homotopic.Quotient as bs :=
  (Quotient.map Path.pi fun x y hxy =>
    Nonempty.map (piHomotopy x y) (Classical.nonempty_pi.mpr hxy)) (Quotient.choice γ)


theorem pi_lift (γ : ∀ i, Path (as i) (bs i)) :
                                                           /-
                                                             ι : Type u_1
                                                             X : ι → Type u_2
                                                             inst✝ : (i : ι) → TopologicalSpace (X i)
                                                             as bs : (i : ι) → X i
                                                             γ : (i : ι) → Path (as i) (bs i)
                                                             ⊢ Eq (Path.Homotopic.pi fun i => Quotient.mk (Path.Homotopic.setoid (as i) (bs …
                                                           -/
    (Path.Homotopic.pi fun i => ⟦γ i⟧) = ⟦Path.pi γ⟧ := by unfold pi; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Composition and products commute.
  This is `Path.trans_pi_eq_pi_trans` descended to path homotopy classes. -/
theorem comp_pi_eq_pi_comp (γ₀ : ∀ i, Path.Homotopic.Quotient (as i) (bs i))
    (γ₁ : ∀ i, Path.Homotopic.Quotient (bs i) (cs i)) : pi γ₀ ⬝ pi γ₁ = pi fun i ↦ γ₀ i ⬝ γ₁ i := by
  induction γ₁ using Quotient.induction_on_pi with | _ a =>
  induction γ₀ using Quotient.induction_on_pi
  simp only [pi_lift]
  rw [← Path.Homotopic.comp_lift, Path.trans_pi_eq_pi_trans, ← pi_lift]
  rfl


/-- Abbreviation for projection onto the ith coordinate. -/
abbrev proj (i : ι) (p : Path.Homotopic.Quotient as bs) : Path.Homotopic.Quotient (as i) (bs i) :=
  p.mapFn ⟨_, continuous_apply i⟩


/-- Lemmas showing projection is the inverse of pi. -/
@[simp]
theorem proj_pi (i : ι) (paths : ∀ i, Path.Homotopic.Quotient (as i) (bs i)) :
    proj i (pi paths) = paths i := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    i : ι
    paths : (i : ι) → Path.Homotopic.Quotient (as i) (bs i)
    ⊢ Eq (Path.Homotopic.proj i (Path.Homotopic.pi paths)) (paths i)
  -/
  induction paths using Quotient.induction_on_pi
  /-
    case h
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    i : ι
    a✝ : (i : ι) → Path (as i) (bs i)
    ⊢ Eq (Path.Homotopic.proj i (Path.Homotopic.pi fun i => Quotient.mk (Path.Homo …
  -/
  rw [proj, pi_lift, ← Path.Homotopic.map_lift]
  /-
    case h
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    i : ι
    a✝ : (i : ι) → Path (as i) (bs i)
    ⊢ Eq (Quotient.mk (Path.Homotopic.setoid ({ toFun := fun p => p i, continuous_ …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem pi_proj (p : Path.Homotopic.Quotient as bs) : (pi fun i => proj i p) = p := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    p : Path.Homotopic.Quotient as bs
    ⊢ Eq (Path.Homotopic.pi fun i => Path.Homotopic.proj i p) p
  -/
  induction p using Quotient.inductionOn
  /-
    case h
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    a✝ : Path as bs
    ⊢ Eq (Path.Homotopic.pi fun i => Path.Homotopic.proj i (Quotient.mk (Path.Homo …
  -/
  simp_rw [proj, ← Path.Homotopic.map_lift]
  /-
    case h
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    a✝ : Path as bs
    ⊢ Eq (Path.Homotopic.pi fun i => Quotient.mk (Path.Homotopic.setoid ({ toFun : …
  -/
  erw [pi_lift]
  /-
    case h
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    as bs : (i : ι) → X i
    a✝ : Path as bs
    ⊢ Eq (Quotient.mk (Path.Homotopic.setoid (fun i => as i) fun i => bs i) (Path. …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The product of homotopies h₁ and h₂.
    This is `HomotopyRel.prod` specialized for path homotopies. -/
def prodHomotopy (h₁ : Path.Homotopy p₁ p₁') (h₂ : Path.Homotopy p₂ p₂') :
    Path.Homotopy (p₁.prod p₂) (p₁'.prod p₂') :=
  ContinuousMap.HomotopyRel.prod h₁ h₂


/-- The product of path classes q₁ and q₂. This is `Path.prod` descended to the quotient. -/
def prod (q₁ : Path.Homotopic.Quotient a₁ a₂) (q₂ : Path.Homotopic.Quotient b₁ b₂) :
    Path.Homotopic.Quotient (a₁, b₁) (a₂, b₂) :=
  Quotient.map₂ Path.prod (fun _ _ h₁ _ _ h₂ => Nonempty.map2 prodHomotopy h₁ h₂) q₁ q₂


theorem prod_lift : prod ⟦p₁⟧ ⟦p₂⟧ = ⟦p₁.prod p₂⟧ :=
  rfl


/-- Products commute with path composition.
    This is `trans_prod_eq_prod_trans` descended to the quotient. -/
theorem comp_prod_eq_prod_comp : prod q₁ q₂ ⬝ prod r₁ r₂ = prod (q₁ ⬝ r₁) (q₂ ⬝ r₂) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ a₃ : α
    b₁ b₂ b₃ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    r₁ : Path.Homotopic.Quotient a₂ a₃
    r₂ : Path.Homotopic.Quotient b₂ b₃
    ⊢ Eq ((Path.Homotopic.prod q₁ q₂).comp (Path.Homotopic.prod r₁ r₂)) (Path.Homo …
  -/
  induction q₁, q₂ using Quotient.inductionOn₂
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ a₃ : α
    b₁ b₂ b₃ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    r₁ : Path.Homotopic.Quotient a₂ a₃
    r₂ : Path.Homotopic.Quotient b₂ b₃
    a✝ : Path a₁ a₂
    b✝ : Path b₁ b₂
    ⊢ Eq ((Path.Homotopic.prod (Quotient.mk (Path.Homotopic.setoid a₁ a₂) a✝) (Quo …
  -/
  induction r₁, r₂ using Quotient.inductionOn₂
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ a₃ : α
    b₁ b₂ b₃ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    r₁ : Path.Homotopic.Quotient a₂ a₃
    r₂ : Path.Homotopic.Quotient b₂ b₃
    a✝¹ : Path a₁ a₂
    b✝¹ : Path b₁ b₂
    a✝ : Path a₂ a₃
    b✝ : Path b₂ b₃
    ⊢ Eq ((Path.Homotopic.prod (Quotient.mk (Path.Homotopic.setoid a₁ a₂) a✝¹) (Qu …
  -/
  simp only [prod_lift, ← Path.Homotopic.comp_lift, Path.trans_prod_eq_prod_trans]
  /-
    🎉 no goals
  -/


/-- Abbreviation for projection onto the left coordinate of a path class. -/
abbrev projLeft (p : Path.Homotopic.Quotient c₁ c₂) : Path.Homotopic.Quotient c₁.1 c₂.1 :=
  p.mapFn ⟨_, continuous_fst⟩


/-- Abbreviation for projection onto the right coordinate of a path class. -/
abbrev projRight (p : Path.Homotopic.Quotient c₁ c₂) : Path.Homotopic.Quotient c₁.2 c₂.2 :=
  p.mapFn ⟨_, continuous_snd⟩


/-- Lemmas showing projection is the inverse of product. -/
@[simp]
theorem projLeft_prod : projLeft (prod q₁ q₂) = q₁ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    ⊢ Eq (Path.Homotopic.projLeft (Path.Homotopic.prod q₁ q₂)) q₁
  -/
  induction q₁, q₂ using Quotient.inductionOn₂
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    a✝ : Path a₁ a₂
    b✝ : Path b₁ b₂
    ⊢ Eq (Path.Homotopic.projLeft (Path.Homotopic.prod (Quotient.mk (Path.Homotopi …
  -/
  rw [projLeft, prod_lift, ← Path.Homotopic.map_lift]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    a✝ : Path a₁ a₂
    b✝ : Path b₁ b₂
    ⊢ Eq (Quotient.mk (Path.Homotopic.setoid ({ toFun := Prod.fst, continuous_toFu …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem projRight_prod : projRight (prod q₁ q₂) = q₂ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    ⊢ Eq (Path.Homotopic.projRight (Path.Homotopic.prod q₁ q₂)) q₂
  -/
  induction q₁, q₂ using Quotient.inductionOn₂
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    a✝ : Path a₁ a₂
    b✝ : Path b₁ b₂
    ⊢ Eq (Path.Homotopic.projRight (Path.Homotopic.prod (Quotient.mk (Path.Homotop …
  -/
  rw [projRight, prod_lift, ← Path.Homotopic.map_lift]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    q₁ : Path.Homotopic.Quotient a₁ a₂
    q₂ : Path.Homotopic.Quotient b₁ b₂
    a✝ : Path a₁ a₂
    b✝ : Path b₁ b₂
    ⊢ Eq (Quotient.mk (Path.Homotopic.setoid ({ toFun := Prod.snd, continuous_toFu …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_projLeft_projRight (p : Path.Homotopic.Quotient (a₁, b₁) (a₂, b₂)) :
    prod (projLeft p) (projRight p) = p := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    p : Path.Homotopic.Quotient { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
    ⊢ Eq (Path.Homotopic.prod (Path.Homotopic.projLeft p) (Path.Homotopic.projRigh …
  -/
  induction p using Quotient.inductionOn
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    a✝ : Path { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
    ⊢ Eq (Path.Homotopic.prod (Path.Homotopic.projLeft (Quotient.mk (Path.Homotopi …
  -/
  simp only [projLeft, projRight, ← Path.Homotopic.map_lift, prod_lift]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    a₁ a₂ : α
    b₁ b₂ : β
    a✝ : Path { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
    ⊢ Eq (Path.Homotopic.prod (Quotient.mk (Path.Homotopic.setoid ({ toFun := Prod …
  -/
  congr
  /-
    🎉 no goals
  -/


