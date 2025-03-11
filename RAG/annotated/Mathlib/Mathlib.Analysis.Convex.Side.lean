/-- The points `x` and `y` are weakly on the same side of `s`. -/
def WSameSide (s : AffineSubspace R P) (x y : P) : Prop :=
  ∃ᵉ (p₁ ∈ s) (p₂ ∈ s), SameRay R (x -ᵥ p₁) (y -ᵥ p₂)


/-- The points `x` and `y` are strictly on the same side of `s`. -/
def SSameSide (s : AffineSubspace R P) (x y : P) : Prop :=
  s.WSameSide x y ∧ x ∉ s ∧ y ∉ s


/-- The points `x` and `y` are weakly on opposite sides of `s`. -/
def WOppSide (s : AffineSubspace R P) (x y : P) : Prop :=
  ∃ᵉ (p₁ ∈ s) (p₂ ∈ s), SameRay R (x -ᵥ p₁) (p₂ -ᵥ y)


/-- The points `x` and `y` are strictly on opposite sides of `s`. -/
def SOppSide (s : AffineSubspace R P) (x y : P) : Prop :=
  s.WOppSide x y ∧ x ∉ s ∧ y ∉ s


theorem WSameSide.map {s : AffineSubspace R P} {x y : P} (h : s.WSameSide x y) (f : P →ᵃ[R] P') :
    (s.map f).WSameSide (f x) (f y) := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    h : s.WSameSide x y
    f : AffineMap R P P'
    ⊢ (AffineSubspace.map f s).WSameSide (f x) (f y)
  -/
  rcases h with ⟨p₁, hp₁, p₂, hp₂, h⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ (AffineSubspace.map f s).WSameSide (f x) (f y)
  -/
  refine ⟨f p₁, mem_map_of_mem f hp₁, f p₂, mem_map_of_mem f hp₂, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f y) (f p₂))
  -/
  simp_rw [← linearMap_vsub]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ SameRay R (f.linear (VSub.vsub x p₁)) (f.linear (VSub.vsub y p₂))
  -/
  exact h.map f.linear
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.wSameSide_map_iff {s : AffineSubspace R P} {x y : P}
    {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    (s.map f).WSameSide (f x) (f y) ↔ s.WSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff ((AffineSubspace.map f s).WSameSide (f x) (f y)) (s.WSameSide x y)
  -/
  refine ⟨fun h => ?_, fun h => h.map _⟩
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    h : (AffineSubspace.map f s).WSameSide (f x) (f y)
    ⊢ s.WSameSide x y
  -/
  rcases h with ⟨fp₁, hfp₁, fp₂, hfp₂, h⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₁ : P'
    hfp₁ : Membership.mem (AffineSubspace.map f s) fp₁
    fp₂ : P'
    hfp₂ : Membership.mem (AffineSubspace.map f s) fp₂
    h : SameRay R (VSub.vsub (f x) fp₁) (VSub.vsub (f y) fp₂)
    ⊢ s.WSameSide x y
  -/
  rw [mem_map] at hfp₁ hfp₂
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₁ : P'
    hfp₁ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₁)
    fp₂ : P'
    hfp₂ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₂)
    h : SameRay R (VSub.vsub (f x) fp₁) (VSub.vsub (f y) fp₂)
    ⊢ s.WSameSide x y
  -/
  rcases hfp₁ with ⟨p₁, hp₁, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₂ : P'
    hfp₂ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₂)
    p₁ : P
    hp₁ : Membership.mem s p₁
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f y) fp₂)
    ⊢ s.WSameSide x y
  -/
  rcases hfp₂ with ⟨p₂, hp₂, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f y) (f p₂))
    ⊢ s.WSameSide x y
  -/
  refine ⟨p₁, hp₁, p₂, hp₂, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f y) (f p₂))
    ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
  -/
  simp_rw [← linearMap_vsub, (f.linear_injective_iff.2 hf).sameRay_map_iff] at h
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.sSameSide_map_iff {s : AffineSubspace R P} {x y : P}
    {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    (s.map f).SSameSide (f x) (f y) ↔ s.SSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff ((AffineSubspace.map f s).SSameSide (f x) (f y)) (s.SSameSide x y)
  -/
  simp_rw [SSameSide, hf.wSameSide_map_iff, mem_map_iff_mem_of_injective hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.AffineEquiv.wSameSide_map_iff {s : AffineSubspace R P} {x y : P} (f : P ≃ᵃ[R] P') :
    (s.map ↑f).WSameSide (f x) (f y) ↔ s.WSameSide x y :=
  (show Function.Injective f.toAffineMap from f.injective).wSameSide_map_iff


@[simp]
theorem _root_.AffineEquiv.sSameSide_map_iff {s : AffineSubspace R P} {x y : P} (f : P ≃ᵃ[R] P') :
    (s.map ↑f).SSameSide (f x) (f y) ↔ s.SSameSide x y :=
  (show Function.Injective f.toAffineMap from f.injective).sSameSide_map_iff


theorem WOppSide.map {s : AffineSubspace R P} {x y : P} (h : s.WOppSide x y) (f : P →ᵃ[R] P') :
    (s.map f).WOppSide (f x) (f y) := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    h : s.WOppSide x y
    f : AffineMap R P P'
    ⊢ (AffineSubspace.map f s).WOppSide (f x) (f y)
  -/
  rcases h with ⟨p₁, hp₁, p₂, hp₂, h⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ (AffineSubspace.map f s).WOppSide (f x) (f y)
  -/
  refine ⟨f p₁, mem_map_of_mem f hp₁, f p₂, mem_map_of_mem f hp₂, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f p₂) (f y))
  -/
  simp_rw [← linearMap_vsub]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ SameRay R (f.linear (VSub.vsub x p₁)) (f.linear (VSub.vsub p₂ y))
  -/
  exact h.map f.linear
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.wOppSide_map_iff {s : AffineSubspace R P} {x y : P}
    {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    (s.map f).WOppSide (f x) (f y) ↔ s.WOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff ((AffineSubspace.map f s).WOppSide (f x) (f y)) (s.WOppSide x y)
  -/
  refine ⟨fun h => ?_, fun h => h.map _⟩
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    h : (AffineSubspace.map f s).WOppSide (f x) (f y)
    ⊢ s.WOppSide x y
  -/
  rcases h with ⟨fp₁, hfp₁, fp₂, hfp₂, h⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₁ : P'
    hfp₁ : Membership.mem (AffineSubspace.map f s) fp₁
    fp₂ : P'
    hfp₂ : Membership.mem (AffineSubspace.map f s) fp₂
    h : SameRay R (VSub.vsub (f x) fp₁) (VSub.vsub fp₂ (f y))
    ⊢ s.WOppSide x y
  -/
  rw [mem_map] at hfp₁ hfp₂
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₁ : P'
    hfp₁ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₁)
    fp₂ : P'
    hfp₂ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₂)
    h : SameRay R (VSub.vsub (f x) fp₁) (VSub.vsub fp₂ (f y))
    ⊢ s.WOppSide x y
  -/
  rcases hfp₁ with ⟨p₁, hp₁, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    fp₂ : P'
    hfp₂ : Exists fun y => And (Membership.mem s y) (Eq (f y) fp₂)
    p₁ : P
    hp₁ : Membership.mem s p₁
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub fp₂ (f y))
    ⊢ s.WOppSide x y
  -/
  rcases hfp₂ with ⟨p₂, hp₂, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f p₂) (f y))
    ⊢ s.WOppSide x y
  -/
  refine ⟨p₁, hp₁, p₂, hp₂, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub (f x) (f p₁)) (VSub.vsub (f p₂) (f y))
    ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
  -/
  simp_rw [← linearMap_vsub, (f.linear_injective_iff.2 hf).sameRay_map_iff] at h
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.sOppSide_map_iff {s : AffineSubspace R P} {x y : P}
    {f : P →ᵃ[R] P'} (hf : Function.Injective f) :
    (s.map f).SOppSide (f x) (f y) ↔ s.SOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    V' : Type u_3
    P : Type u_4
    P' : Type u_5
    inst✝⁶ : StrictOrderedCommRing R
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module R V
    inst✝³ : AddTorsor V P
    inst✝² : AddCommGroup V'
    inst✝¹ : Module R V'
    inst✝ : AddTorsor V' P'
    s : AffineSubspace R P
    x y : P
    f : AffineMap R P P'
    hf : Function.Injective ⇑f
    ⊢ Iff ((AffineSubspace.map f s).SOppSide (f x) (f y)) (s.SOppSide x y)
  -/
  simp_rw [SOppSide, hf.wOppSide_map_iff, mem_map_iff_mem_of_injective hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.AffineEquiv.wOppSide_map_iff {s : AffineSubspace R P} {x y : P} (f : P ≃ᵃ[R] P') :
    (s.map ↑f).WOppSide (f x) (f y) ↔ s.WOppSide x y :=
  (show Function.Injective f.toAffineMap from f.injective).wOppSide_map_iff


@[simp]
theorem _root_.AffineEquiv.sOppSide_map_iff {s : AffineSubspace R P} {x y : P} (f : P ≃ᵃ[R] P') :
    (s.map ↑f).SOppSide (f x) (f y) ↔ s.SOppSide x y :=
  (show Function.Injective f.toAffineMap from f.injective).sOppSide_map_iff


theorem WSameSide.nonempty {s : AffineSubspace R P} {x y : P} (h : s.WSameSide x y) :
    (s : Set P).Nonempty :=
  ⟨h.choose, h.choose_spec.left⟩


theorem SSameSide.nonempty {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) :
    (s : Set P).Nonempty :=
  ⟨h.1.choose, h.1.choose_spec.left⟩


theorem WOppSide.nonempty {s : AffineSubspace R P} {x y : P} (h : s.WOppSide x y) :
    (s : Set P).Nonempty :=
  ⟨h.choose, h.choose_spec.left⟩


theorem SOppSide.nonempty {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) :
    (s : Set P).Nonempty :=
  ⟨h.1.choose, h.1.choose_spec.left⟩


theorem SSameSide.wSameSide {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) :
    s.WSameSide x y :=
  h.1


theorem SSameSide.left_not_mem {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) : x ∉ s :=
  h.2.1


theorem SSameSide.right_not_mem {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) : y ∉ s :=
  h.2.2


theorem SOppSide.wOppSide {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) :
    s.WOppSide x y :=
  h.1


theorem SOppSide.left_not_mem {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) : x ∉ s :=
  h.2.1


theorem SOppSide.right_not_mem {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) : y ∉ s :=
  h.2.2


theorem wSameSide_comm {s : AffineSubspace R P} {x y : P} : s.WSameSide x y ↔ s.WSameSide y x :=
  ⟨fun ⟨p₁, hp₁, p₂, hp₂, h⟩ => ⟨p₂, hp₂, p₁, hp₁, h.symm⟩,
    fun ⟨p₁, hp₁, p₂, hp₂, h⟩ => ⟨p₂, hp₂, p₁, hp₁, h.symm⟩⟩


alias ⟨WSameSide.symm, _⟩ := wSameSide_comm


theorem sSameSide_comm {s : AffineSubspace R P} {x y : P} : s.SSameSide x y ↔ s.SSameSide y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    ⊢ Iff (s.SSameSide x y) (s.SSameSide y x)
  -/
  rw [SSameSide, SSameSide, wSameSide_comm, and_comm (b := x ∉ s)]
  /-
    🎉 no goals
  -/


alias ⟨SSameSide.symm, _⟩ := sSameSide_comm


theorem wOppSide_comm {s : AffineSubspace R P} {x y : P} : s.WOppSide x y ↔ s.WOppSide y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    ⊢ Iff (s.WOppSide x y) (s.WOppSide y x)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      ⊢ s.WOppSide x y → s.WOppSide y x
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
      ⊢ s.WOppSide y x
    -/
    refine ⟨p₂, hp₂, p₁, hp₁, ?_⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
      ⊢ SameRay R (VSub.vsub y p₂) (VSub.vsub p₁ x)
    -/
    rwa [SameRay.sameRay_comm, ← sameRay_neg_iff, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      ⊢ s.WOppSide y x → s.WOppSide x y
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub y p₁) (VSub.vsub p₂ x)
      ⊢ s.WOppSide x y
    -/
    refine ⟨p₂, hp₂, p₁, hp₁, ?_⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub y p₁) (VSub.vsub p₂ x)
      ⊢ SameRay R (VSub.vsub x p₂) (VSub.vsub p₁ y)
    -/
    rwa [SameRay.sameRay_comm, ← sameRay_neg_iff, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev]
    /-
      🎉 no goals
    -/


alias ⟨WOppSide.symm, _⟩ := wOppSide_comm


theorem sOppSide_comm {s : AffineSubspace R P} {x y : P} : s.SOppSide x y ↔ s.SOppSide y x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    ⊢ Iff (s.SOppSide x y) (s.SOppSide y x)
  -/
  rw [SOppSide, SOppSide, wOppSide_comm, and_comm (b := x ∉ s)]
  /-
    🎉 no goals
  -/


alias ⟨SOppSide.symm, _⟩ := sOppSide_comm


theorem not_wSameSide_bot (x y : P) : ¬(⊥ : AffineSubspace R P).WSameSide x y :=
  fun ⟨_, h, _⟩ => h.elim


theorem not_sSameSide_bot (x y : P) : ¬(⊥ : AffineSubspace R P).SSameSide x y :=
  fun h => not_wSameSide_bot x y h.wSameSide


theorem not_wOppSide_bot (x y : P) : ¬(⊥ : AffineSubspace R P).WOppSide x y :=
  fun ⟨_, h, _⟩ => h.elim


theorem not_sOppSide_bot (x y : P) : ¬(⊥ : AffineSubspace R P).SOppSide x y :=
  fun h => not_wOppSide_bot x y h.wOppSide


@[simp]
theorem wSameSide_self_iff {s : AffineSubspace R P} {x : P} :
    s.WSameSide x x ↔ (s : Set P).Nonempty :=
  ⟨fun h => h.nonempty, fun ⟨p, hp⟩ => ⟨p, hp, p, hp, SameRay.rfl⟩⟩


theorem sSameSide_self_iff {s : AffineSubspace R P} {x : P} :
    s.SSameSide x x ↔ (s : Set P).Nonempty ∧ x ∉ s :=
  ⟨fun ⟨h, hx, _⟩ => ⟨wSameSide_self_iff.1 h, hx⟩, fun ⟨h, hx⟩ => ⟨wSameSide_self_iff.2 h, hx, hx⟩⟩


theorem wSameSide_of_left_mem {s : AffineSubspace R P} {x : P} (y : P) (hx : x ∈ s) :
    s.WSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ s.WSameSide x y
  -/
  refine ⟨x, hx, x, hx, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ SameRay R (VSub.vsub x x) (VSub.vsub y x)
  -/
  rw [vsub_self]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ SameRay R 0 (VSub.vsub y x)
  -/
  apply SameRay.zero_left
  /-
    🎉 no goals
  -/


theorem wSameSide_of_right_mem {s : AffineSubspace R P} (x : P) {y : P} (hy : y ∈ s) :
    s.WSameSide x y :=
  (wSameSide_of_left_mem x hy).symm


theorem wOppSide_of_left_mem {s : AffineSubspace R P} {x : P} (y : P) (hx : x ∈ s) :
    s.WOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ s.WOppSide x y
  -/
  refine ⟨x, hx, x, hx, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ SameRay R (VSub.vsub x x) (VSub.vsub x y)
  -/
  rw [vsub_self]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    ⊢ SameRay R 0 (VSub.vsub x y)
  -/
  apply SameRay.zero_left
  /-
    🎉 no goals
  -/


theorem wOppSide_of_right_mem {s : AffineSubspace R P} (x : P) {y : P} (hy : y ∈ s) :
    s.WOppSide x y :=
  (wOppSide_of_left_mem x hy).symm


theorem wSameSide_vadd_left_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.WSameSide (v +ᵥ x) y ↔ s.WSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.WSameSide (HVAdd.hVAdd v x) y) (s.WSameSide x y)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      ⊢ s.WSameSide (HVAdd.hVAdd v x) y → s.WSameSide x y
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    refine
      ⟨-v +ᵥ p₁, AffineSubspace.vadd_mem_of_mem_direction (Submodule.neg_mem _ hv) hp₁, p₂, hp₂, ?_⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub (HVAdd.hVAdd v x) p₁) (VSub.vsub y p₂)
      ⊢ SameRay R (VSub.vsub x (HVAdd.hVAdd (Neg.neg v) p₁)) (VSub.vsub y p₂)
    -/
    rwa [vsub_vadd_eq_vsub_sub, sub_neg_eq_add, add_comm, ← vadd_vsub_assoc]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      ⊢ s.WSameSide x y → s.WSameSide (HVAdd.hVAdd v x) y
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
      ⊢ s.WSameSide (HVAdd.hVAdd v x) y
    -/
    refine ⟨v +ᵥ p₁, AffineSubspace.vadd_mem_of_mem_direction hv hp₁, p₂, hp₂, ?_⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
      ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd v x) (HVAdd.hVAdd v p₁)) (VSub.vsub y p₂)
    -/
    rwa [vadd_vsub_vadd_cancel_left]
    /-
      🎉 no goals
    -/


theorem wSameSide_vadd_right_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.WSameSide x (v +ᵥ y) ↔ s.WSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.WSameSide x (HVAdd.hVAdd v y)) (s.WSameSide x y)
  -/
  rw [wSameSide_comm, wSameSide_vadd_left_iff hv, wSameSide_comm]
  /-
    🎉 no goals
  -/


theorem sSameSide_vadd_left_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.SSameSide (v +ᵥ x) y ↔ s.SSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.SSameSide (HVAdd.hVAdd v x) y) (s.SSameSide x y)
  -/
  rw [SSameSide, SSameSide, wSameSide_vadd_left_iff hv, vadd_mem_iff_mem_of_mem_direction hv]
  /-
    🎉 no goals
  -/


theorem sSameSide_vadd_right_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.SSameSide x (v +ᵥ y) ↔ s.SSameSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.SSameSide x (HVAdd.hVAdd v y)) (s.SSameSide x y)
  -/
  rw [sSameSide_comm, sSameSide_vadd_left_iff hv, sSameSide_comm]
  /-
    🎉 no goals
  -/


theorem wOppSide_vadd_left_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.WOppSide (v +ᵥ x) y ↔ s.WOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.WOppSide (HVAdd.hVAdd v x) y) (s.WOppSide x y)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      ⊢ s.WOppSide (HVAdd.hVAdd v x) y → s.WOppSide x y
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    refine
      ⟨-v +ᵥ p₁, AffineSubspace.vadd_mem_of_mem_direction (Submodule.neg_mem _ hv) hp₁, p₂, hp₂, ?_⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub (HVAdd.hVAdd v x) p₁) (VSub.vsub p₂ y)
      ⊢ SameRay R (VSub.vsub x (HVAdd.hVAdd (Neg.neg v) p₁)) (VSub.vsub p₂ y)
    -/
    rwa [vsub_vadd_eq_vsub_sub, sub_neg_eq_add, add_comm, ← vadd_vsub_assoc]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      ⊢ s.WOppSide x y → s.WOppSide (HVAdd.hVAdd v x) y
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
      ⊢ s.WOppSide (HVAdd.hVAdd v x) y
    -/
    refine ⟨v +ᵥ p₁, AffineSubspace.vadd_mem_of_mem_direction hv hp₁, p₂, hp₂, ?_⟩
    /-
      case mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      v : V
      hv : Membership.mem s.direction v
      p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
      ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd v x) (HVAdd.hVAdd v p₁)) (VSub.vsub p₂ y)
    -/
    rwa [vadd_vsub_vadd_cancel_left]
    /-
      🎉 no goals
    -/


theorem wOppSide_vadd_right_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.WOppSide x (v +ᵥ y) ↔ s.WOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.WOppSide x (HVAdd.hVAdd v y)) (s.WOppSide x y)
  -/
  rw [wOppSide_comm, wOppSide_vadd_left_iff hv, wOppSide_comm]
  /-
    🎉 no goals
  -/


theorem sOppSide_vadd_left_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.SOppSide (v +ᵥ x) y ↔ s.SOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.SOppSide (HVAdd.hVAdd v x) y) (s.SOppSide x y)
  -/
  rw [SOppSide, SOppSide, wOppSide_vadd_left_iff hv, vadd_mem_iff_mem_of_mem_direction hv]
  /-
    🎉 no goals
  -/


theorem sOppSide_vadd_right_iff {s : AffineSubspace R P} {x y : P} {v : V} (hv : v ∈ s.direction) :
    s.SOppSide x (v +ᵥ y) ↔ s.SOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    v : V
    hv : Membership.mem s.direction v
    ⊢ Iff (s.SOppSide x (HVAdd.hVAdd v y)) (s.SOppSide x y)
  -/
  rw [sOppSide_comm, sOppSide_vadd_left_iff hv, sOppSide_comm]
  /-
    🎉 no goals
  -/


theorem wSameSide_smul_vsub_vadd_left {s : AffineSubspace R P} {p₁ p₂ : P} (x : P) (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) {t : R} (ht : 0 ≤ t) : s.WSameSide (t • (x -ᵥ p₁) +ᵥ p₂) x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le 0 t
    ⊢ s.WSameSide (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) x
  -/
  refine ⟨p₂, hp₂, p₁, hp₁, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le 0 t
    ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) p₂) ( …
  -/
  rw [vadd_vsub]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le 0 t
    ⊢ SameRay R (HSMul.hSMul t (VSub.vsub x p₁)) (VSub.vsub x p₁)
  -/
  exact SameRay.sameRay_nonneg_smul_left _ ht
  /-
    🎉 no goals
  -/


theorem wSameSide_smul_vsub_vadd_right {s : AffineSubspace R P} {p₁ p₂ : P} (x : P) (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) {t : R} (ht : 0 ≤ t) : s.WSameSide x (t • (x -ᵥ p₁) +ᵥ p₂) :=
  (wSameSide_smul_vsub_vadd_left x hp₁ hp₂ ht).symm


theorem wSameSide_lineMap_left {s : AffineSubspace R P} {x : P} (y : P) (h : x ∈ s) {t : R}
    (ht : 0 ≤ t) : s.WSameSide (lineMap x y t) y :=
  wSameSide_smul_vsub_vadd_left y h h ht


theorem wSameSide_lineMap_right {s : AffineSubspace R P} {x : P} (y : P) (h : x ∈ s) {t : R}
    (ht : 0 ≤ t) : s.WSameSide y (lineMap x y t) :=
  (wSameSide_lineMap_left y h ht).symm


theorem wOppSide_smul_vsub_vadd_left {s : AffineSubspace R P} {p₁ p₂ : P} (x : P) (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) {t : R} (ht : t ≤ 0) : s.WOppSide (t • (x -ᵥ p₁) +ᵥ p₂) x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le t 0
    ⊢ s.WOppSide (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) x
  -/
  refine ⟨p₂, hp₂, p₁, hp₁, ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le t 0
    ⊢ SameRay R (VSub.vsub (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) p₂) ( …
  -/
  rw [vadd_vsub, ← neg_neg t, neg_smul, ← smul_neg, neg_vsub_eq_vsub_rev]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    p₁ p₂ x : P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LE.le t 0
    ⊢ SameRay R (HSMul.hSMul (Neg.neg t) (VSub.vsub p₁ x)) (VSub.vsub p₁ x)
  -/
  exact SameRay.sameRay_nonneg_smul_left _ (neg_nonneg.2 ht)
  /-
    🎉 no goals
  -/


theorem wOppSide_smul_vsub_vadd_right {s : AffineSubspace R P} {p₁ p₂ : P} (x : P) (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) {t : R} (ht : t ≤ 0) : s.WOppSide x (t • (x -ᵥ p₁) +ᵥ p₂) :=
  (wOppSide_smul_vsub_vadd_left x hp₁ hp₂ ht).symm


theorem wOppSide_lineMap_left {s : AffineSubspace R P} {x : P} (y : P) (h : x ∈ s) {t : R}
    (ht : t ≤ 0) : s.WOppSide (lineMap x y t) y :=
  wOppSide_smul_vsub_vadd_left y h h ht


theorem wOppSide_lineMap_right {s : AffineSubspace R P} {x : P} (y : P) (h : x ∈ s) {t : R}
    (ht : t ≤ 0) : s.WOppSide y (lineMap x y t) :=
  (wOppSide_lineMap_left y h ht).symm


theorem _root_.Wbtw.wSameSide₂₃ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hx : x ∈ s) : s.WSameSide y z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    h : Wbtw R x y z
    hx : Membership.mem s x
    ⊢ s.WSameSide y z
  -/
  rcases h with ⟨t, ⟨ht0, -⟩, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    hx : Membership.mem s x
    t : R
    ht0 : LE.le 0 t
    ⊢ s.WSameSide ((AffineMap.lineMap x z) t) z
  -/
  exact wSameSide_lineMap_left z hx ht0
  /-
    🎉 no goals
  -/


theorem _root_.Wbtw.wSameSide₃₂ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hx : x ∈ s) : s.WSameSide z y :=
  (h.wSameSide₂₃ hx).symm


theorem _root_.Wbtw.wSameSide₁₂ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hz : z ∈ s) : s.WSameSide x y :=
  h.symm.wSameSide₃₂ hz


theorem _root_.Wbtw.wSameSide₂₁ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hz : z ∈ s) : s.WSameSide y x :=
  h.symm.wSameSide₂₃ hz


theorem _root_.Wbtw.wOppSide₁₃ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hy : y ∈ s) : s.WOppSide x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    h : Wbtw R x y z
    hy : Membership.mem s y
    ⊢ s.WOppSide x z
  -/
  rcases h with ⟨t, ⟨ht0, ht1⟩, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ⊢ s.WOppSide x z
  -/
  refine ⟨_, hy, _, hy, ?_⟩
  /-
    case intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ⊢ SameRay R (VSub.vsub x ((AffineMap.lineMap x z) t)) (VSub.vsub ((AffineMap.l …
  -/
  rcases ht1.lt_or_eq with (ht1' | rfl); swap
    /-
      case intro.intro.intro.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x z : P
      ht0 : LE.le 0 1
      ht1 : LE.le 1 1
      hy : Membership.mem s ((AffineMap.lineMap x z) 1)
      ⊢ SameRay R (VSub.vsub x ((AffineMap.lineMap x z) 1)) (VSub.vsub ((AffineMap.l …
    -/
  · rw [lineMap_apply_one]; simp
                            /-
                              🎉 no goals
                            -/
  /-
    case intro.intro.intro.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ht1' : LT.lt t 1
    ⊢ SameRay R (VSub.vsub x ((AffineMap.lineMap x z) t)) (VSub.vsub ((AffineMap.l …
  -/
  rcases ht0.lt_or_eq with (ht0' | rfl); swap
    /-
      case intro.intro.intro.inl.inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : StrictOrderedCommRing R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x z : P
      ht0 : LE.le 0 0
      ht1 : LE.le 0 1
      hy : Membership.mem s ((AffineMap.lineMap x z) 0)
      ht1' : LT.lt 0 1
      ⊢ SameRay R (VSub.vsub x ((AffineMap.lineMap x z) 0)) (VSub.vsub ((AffineMap.l …
    -/
  · rw [lineMap_apply_zero]; simp
                             /-
                               🎉 no goals
                             -/
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ht1' : LT.lt t 1
    ht0' : LT.lt 0 t
    ⊢ SameRay R (VSub.vsub x ((AffineMap.lineMap x z) t)) (VSub.vsub ((AffineMap.l …
  -/
  refine Or.inr (Or.inr ⟨1 - t, t, sub_pos.2 ht1', ht0', ?_⟩)
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ht1' : LT.lt t 1
    ht0' : LT.lt 0 t
    ⊢ Eq (HSMul.hSMul (HSub.hSub 1 t) (VSub.vsub x ((AffineMap.lineMap x z) t))) ( …
  -/
  rw [lineMap_apply, vadd_vsub_assoc, vsub_vadd_eq_vsub_sub, ← neg_vsub_eq_vsub_rev z, vsub_self]
  /-
    case intro.intro.intro.inl.inl
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : StrictOrderedCommRing R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    ht1' : LT.lt t 1
    ht0' : LT.lt 0 t
    ⊢ Eq (HSMul.hSMul (HSub.hSub 1 t) (HSub.hSub 0 (HSMul.hSMul t (VSub.vsub z x)) …
  -/
  module
  /-
    🎉 no goals
  -/


theorem _root_.Wbtw.wOppSide₃₁ {s : AffineSubspace R P} {x y z : P} (h : Wbtw R x y z)
    (hy : y ∈ s) : s.WOppSide z x :=
  h.symm.wOppSide₁₃ hy


@[simp]
theorem wOppSide_self_iff {s : AffineSubspace R P} {x : P} : s.WOppSide x x ↔ x ∈ s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x : P
    ⊢ Iff (s.WOppSide x x) (Membership.mem s x)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x : P
      ⊢ s.WOppSide x x → Membership.mem s x
    -/
  · rintro ⟨p₁, hp₁, p₂, hp₂, h⟩
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ x)
      ⊢ Membership.mem s x
    -/
    obtain ⟨a, -, -, -, -, h₁, -⟩ := h.exists_eq_smul_add
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ x)
      a : R
      h₁ : Eq (VSub.vsub x p₁) (HSMul.hSMul a (HAdd.hAdd (VSub.vsub x p₁) (VSub.vsub …
      ⊢ Membership.mem s x
    -/
    rw [add_comm, vsub_add_vsub_cancel, ← eq_vadd_iff_vsub_eq] at h₁
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ x)
      a : R
      h₁ : Eq x (HVAdd.hVAdd (HSMul.hSMul a (VSub.vsub p₂ p₁)) p₁)
      ⊢ Membership.mem s x
    -/
    rw [h₁]
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ x)
      a : R
      h₁ : Eq x (HVAdd.hVAdd (HSMul.hSMul a (VSub.vsub p₂ p₁)) p₁)
      ⊢ Membership.mem s (HVAdd.hVAdd (HSMul.hSMul a (VSub.vsub p₂ p₁)) p₁)
    -/
    exact s.smul_vsub_vadd_mem a hp₂ hp₁ hp₁
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x : P
      ⊢ Membership.mem s x → s.WOppSide x x
    -/
  · exact fun h => ⟨x, h, x, h, SameRay.rfl⟩
    /-
      🎉 no goals
    -/


theorem not_sOppSide_self (s : AffineSubspace R P) (x : P) : ¬s.SOppSide x x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x : P
    ⊢ Not (s.SOppSide x x)
  -/
  rw [SOppSide]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x : P
    ⊢ Not (And (s.WOppSide x x) (And (Not (Membership.mem s x)) (Not (Membership.m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem wSameSide_iff_exists_left {s : AffineSubspace R P} {x y p₁ : P} (h : p₁ ∈ s) :
    s.WSameSide x y ↔ x ∈ s ∨ ∃ p₂ ∈ s, SameRay R (x -ᵥ p₁) (y -ᵥ p₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Iff (s.WSameSide x y) (Or (Membership.mem s x) (Exists fun p₂ => And (Member …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      h : Membership.mem s p₁
      ⊢ s.WSameSide x y → Or (Membership.mem s x) (Exists fun p₂ => And (Membership. …
    -/
  · rintro ⟨p₁', hp₁', p₂', hp₂', h0 | h0 | ⟨r₁, r₂, hr₁, hr₂, hr⟩⟩
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub x p₁') 0
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
    · rw [vsub_eq_zero_iff_eq] at h0
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq x p₁'
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
      rw [h0]
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq x p₁'
        ⊢ Or (Membership.mem s p₁') (Exists fun p₂ => And (Membership.mem s p₂) (SameR …
      -/
      exact Or.inl hp₁'
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub y p₂') 0
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
    · refine Or.inr ⟨p₂', hp₂', ?_⟩
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub y p₂') 0
        ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂')
      -/
      rw [h0]
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub y p₂') 0
        ⊢ SameRay R (VSub.vsub x p₁) 0
      -/
      exact SameRay.zero_right _
      /-
        🎉 no goals
      -/
    · refine Or.inr ⟨(r₁ / r₂) • (p₁ -ᵥ p₁') +ᵥ p₂', s.smul_vsub_vadd_mem _ h hp₁' hp₂',
        Or.inr (Or.inr ⟨r₁, r₂, hr₁, hr₂, ?_⟩)⟩
      rw [vsub_vadd_eq_vsub_sub, smul_sub, ← hr, smul_smul, mul_div_cancel₀ _ hr₂.ne.symm,
        ← smul_sub, vsub_sub_vsub_cancel_right]
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      h : Membership.mem s p₁
      ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
    -/
  · rintro (h' | ⟨h₁, h₂, h₃⟩)
      /-
        case mpr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        h' : Membership.mem s x
        ⊢ s.WSameSide x y
      -/
    · exact wSameSide_of_left_mem y h'
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        h₁ : P
        h₂ : Membership.mem s h₁
        h₃ : SameRay R (VSub.vsub x p₁) (VSub.vsub y h₁)
        ⊢ s.WSameSide x y
      -/
    · exact ⟨p₁, h, h₁, h₂, h₃⟩
      /-
        🎉 no goals
      -/


theorem wSameSide_iff_exists_right {s : AffineSubspace R P} {x y p₂ : P} (h : p₂ ∈ s) :
    s.WSameSide x y ↔ y ∈ s ∨ ∃ p₁ ∈ s, SameRay R (x -ᵥ p₁) (y -ᵥ p₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (s.WSameSide x y) (Or (Membership.mem s y) (Exists fun p₁ => And (Member …
  -/
  rw [wSameSide_comm, wSameSide_iff_exists_left h]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (Or (Membership.mem s y) (Exists fun p₂_1 => And (Membership.mem s p₂_1) …
  -/
  simp_rw [SameRay.sameRay_comm]
  /-
    🎉 no goals
  -/


theorem sSameSide_iff_exists_left {s : AffineSubspace R P} {x y p₁ : P} (h : p₁ ∈ s) :
    s.SSameSide x y ↔ x ∉ s ∧ y ∉ s ∧ ∃ p₂ ∈ s, SameRay R (x -ᵥ p₁) (y -ᵥ p₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Iff (s.SSameSide x y) (And (Not (Membership.mem s x)) (And (Not (Membership. …
  -/
  rw [SSameSide, and_comm, wSameSide_iff_exists_left h, and_assoc, and_congr_right_iff]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Not (Membership.mem s x) → Iff (And (Not (Membership.mem s y)) (Or (Membersh …
  -/
  intro hx
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    hx : Not (Membership.mem s x)
    ⊢ Iff (And (Not (Membership.mem s y)) (Or (Membership.mem s x) (Exists fun p₂  …
  -/
  rw [or_iff_right hx]
  /-
    🎉 no goals
  -/


theorem sSameSide_iff_exists_right {s : AffineSubspace R P} {x y p₂ : P} (h : p₂ ∈ s) :
    s.SSameSide x y ↔ x ∉ s ∧ y ∉ s ∧ ∃ p₁ ∈ s, SameRay R (x -ᵥ p₁) (y -ᵥ p₂) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (s.SSameSide x y) (And (Not (Membership.mem s x)) (And (Not (Membership. …
  -/
  rw [sSameSide_comm, sSameSide_iff_exists_left h, ← and_assoc, and_comm (a := y ∉ s), and_assoc]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (And (Not (Membership.mem s x)) (And (Not (Membership.mem s y)) (Exists  …
  -/
  simp_rw [SameRay.sameRay_comm]
  /-
    🎉 no goals
  -/


theorem wOppSide_iff_exists_left {s : AffineSubspace R P} {x y p₁ : P} (h : p₁ ∈ s) :
    s.WOppSide x y ↔ x ∈ s ∨ ∃ p₂ ∈ s, SameRay R (x -ᵥ p₁) (p₂ -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Iff (s.WOppSide x y) (Or (Membership.mem s x) (Exists fun p₂ => And (Members …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      h : Membership.mem s p₁
      ⊢ s.WOppSide x y → Or (Membership.mem s x) (Exists fun p₂ => And (Membership.m …
    -/
  · rintro ⟨p₁', hp₁', p₂', hp₂', h0 | h0 | ⟨r₁, r₂, hr₁, hr₂, hr⟩⟩
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub x p₁') 0
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
    · rw [vsub_eq_zero_iff_eq] at h0
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq x p₁'
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
      rw [h0]
      /-
        case mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq x p₁'
        ⊢ Or (Membership.mem s p₁') (Exists fun p₂ => And (Membership.mem s p₂) (SameR …
      -/
      exact Or.inl hp₁'
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub p₂' y) 0
        ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
      -/
    · refine Or.inr ⟨p₂', hp₂', ?_⟩
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub p₂' y) 0
        ⊢ SameRay R (VSub.vsub x p₁) (VSub.vsub p₂' y)
      -/
      rw [h0]
      /-
        case mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        h0 : Eq (VSub.vsub p₂' y) 0
        ⊢ SameRay R (VSub.vsub x p₁) 0
      -/
      exact SameRay.zero_right _
      /-
        🎉 no goals
      -/
    · refine Or.inr ⟨(-r₁ / r₂) • (p₁ -ᵥ p₁') +ᵥ p₂', s.smul_vsub_vadd_mem _ h hp₁' hp₂',
        Or.inr (Or.inr ⟨r₁, r₂, hr₁, hr₂, ?_⟩)⟩
      /-
        case mp.intro.intro.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        hr : Eq (HSMul.hSMul r₁ (VSub.vsub x p₁')) (HSMul.hSMul r₂ (VSub.vsub p₂' y))
        ⊢ Eq (HSMul.hSMul r₁ (VSub.vsub x p₁)) (HSMul.hSMul r₂ (VSub.vsub (HVAdd.hVAdd …
      -/
      rw [vadd_vsub_assoc, ← vsub_sub_vsub_cancel_right x p₁ p₁']
      /-
        case mp.intro.intro.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        hr : Eq (HSMul.hSMul r₁ (VSub.vsub x p₁')) (HSMul.hSMul r₂ (VSub.vsub p₂' y))
        ⊢ Eq (HSMul.hSMul r₁ (HSub.hSub (VSub.vsub x p₁') (VSub.vsub p₁ p₁'))) (HSMul. …
      -/
      linear_combination (norm := match_scalars <;> field_simp) hr
      /-
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        p₁' : P
        hp₁' : Membership.mem s p₁'
        p₂' : P
        hp₂' : Membership.mem s p₂'
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        hr : Eq (HSMul.hSMul r₁ (VSub.vsub x p₁')) (HSMul.hSMul r₂ (VSub.vsub p₂' y))
        ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul r₁ r₂)) (HMul.hMul r₂ r₁)) 0
      -/
      ring
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      h : Membership.mem s p₁
      ⊢ Or (Membership.mem s x) (Exists fun p₂ => And (Membership.mem s p₂) (SameRay …
    -/
  · rintro (h' | ⟨h₁, h₂, h₃⟩)
      /-
        case mpr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        h' : Membership.mem s x
        ⊢ s.WOppSide x y
      -/
    · exact wOppSide_of_left_mem y h'
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        h : Membership.mem s p₁
        h₁ : P
        h₂ : Membership.mem s h₁
        h₃ : SameRay R (VSub.vsub x p₁) (VSub.vsub h₁ y)
        ⊢ s.WOppSide x y
      -/
    · exact ⟨p₁, h, h₁, h₂, h₃⟩
      /-
        🎉 no goals
      -/


theorem wOppSide_iff_exists_right {s : AffineSubspace R P} {x y p₂ : P} (h : p₂ ∈ s) :
    s.WOppSide x y ↔ y ∈ s ∨ ∃ p₁ ∈ s, SameRay R (x -ᵥ p₁) (p₂ -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (s.WOppSide x y) (Or (Membership.mem s y) (Exists fun p₁ => And (Members …
  -/
  rw [wOppSide_comm, wOppSide_iff_exists_left h]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Iff (Or (Membership.mem s y) (Exists fun p₂_1 => And (Membership.mem s p₂_1) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      ⊢ Or (Membership.mem s y) (Exists fun p₂_1 => And (Membership.mem s p₂_1) (Sam …
    -/
  · rintro (hy | ⟨p, hp, hr⟩)
      /-
        case mp.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₂ : P
        h : Membership.mem s p₂
        hy : Membership.mem s y
        ⊢ Or (Membership.mem s y) (Exists fun p₁ => And (Membership.mem s p₁) (SameRay …
      -/
    · exact Or.inl hy
      /-
        🎉 no goals
      -/
    /-
      case mp.inr.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      p : P
      hp : Membership.mem s p
      hr : SameRay R (VSub.vsub y p₂) (VSub.vsub p x)
      ⊢ Or (Membership.mem s y) (Exists fun p₁ => And (Membership.mem s p₁) (SameRay …
    -/
    refine Or.inr ⟨p, hp, ?_⟩
    /-
      case mp.inr.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      p : P
      hp : Membership.mem s p
      hr : SameRay R (VSub.vsub y p₂) (VSub.vsub p x)
      ⊢ SameRay R (VSub.vsub x p) (VSub.vsub p₂ y)
    -/
    rwa [SameRay.sameRay_comm, ← sameRay_neg_iff, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      ⊢ Or (Membership.mem s y) (Exists fun p₁ => And (Membership.mem s p₁) (SameRay …
    -/
  · rintro (hy | ⟨p, hp, hr⟩)
      /-
        case mpr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₂ : P
        h : Membership.mem s p₂
        hy : Membership.mem s y
        ⊢ Or (Membership.mem s y) (Exists fun p₂_1 => And (Membership.mem s p₂_1) (Sam …
      -/
    · exact Or.inl hy
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      p : P
      hp : Membership.mem s p
      hr : SameRay R (VSub.vsub x p) (VSub.vsub p₂ y)
      ⊢ Or (Membership.mem s y) (Exists fun p₂_1 => And (Membership.mem s p₂_1) (Sam …
    -/
    refine Or.inr ⟨p, hp, ?_⟩
    /-
      case mpr.inr.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₂ : P
      h : Membership.mem s p₂
      p : P
      hp : Membership.mem s p
      hr : SameRay R (VSub.vsub x p) (VSub.vsub p₂ y)
      ⊢ SameRay R (VSub.vsub y p₂) (VSub.vsub p x)
    -/
    rwa [SameRay.sameRay_comm, ← sameRay_neg_iff, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev]
    /-
      🎉 no goals
    -/


theorem sOppSide_iff_exists_left {s : AffineSubspace R P} {x y p₁ : P} (h : p₁ ∈ s) :
    s.SOppSide x y ↔ x ∉ s ∧ y ∉ s ∧ ∃ p₂ ∈ s, SameRay R (x -ᵥ p₁) (p₂ -ᵥ y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Iff (s.SOppSide x y) (And (Not (Membership.mem s x)) (And (Not (Membership.m …
  -/
  rw [SOppSide, and_comm, wOppSide_iff_exists_left h, and_assoc, and_congr_right_iff]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    ⊢ Not (Membership.mem s x) → Iff (And (Not (Membership.mem s y)) (Or (Membersh …
  -/
  intro hx
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₁ : P
    h : Membership.mem s p₁
    hx : Not (Membership.mem s x)
    ⊢ Iff (And (Not (Membership.mem s y)) (Or (Membership.mem s x) (Exists fun p₂  …
  -/
  rw [or_iff_right hx]
  /-
    🎉 no goals
  -/


theorem sOppSide_iff_exists_right {s : AffineSubspace R P} {x y p₂ : P} (h : p₂ ∈ s) :
    s.SOppSide x y ↔ x ∉ s ∧ y ∉ s ∧ ∃ p₁ ∈ s, SameRay R (x -ᵥ p₁) (p₂ -ᵥ y) := by
  rw [SOppSide, and_comm, wOppSide_iff_exists_right h, and_assoc, and_congr_right_iff,
    and_congr_right_iff]
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    ⊢ Not (Membership.mem s x) → Not (Membership.mem s y) → Iff (Or (Membership.me …
  -/
  rintro _ hy
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y p₂ : P
    h : Membership.mem s p₂
    a✝ : Not (Membership.mem s x)
    hy : Not (Membership.mem s y)
    ⊢ Iff (Or (Membership.mem s y) (Exists fun p₁ => And (Membership.mem s p₁) (Sa …
  -/
  rw [or_iff_right hy]
  /-
    🎉 no goals
  -/


theorem WSameSide.trans {s : AffineSubspace R P} {x y z : P} (hxy : s.WSameSide x y)
    (hyz : s.WSameSide y z) (hy : y ∉ s) : s.WSameSide x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hxy : s.WSameSide x y
    hyz : s.WSameSide y z
    hy : Not (Membership.mem s y)
    ⊢ s.WSameSide x z
  -/
  rcases hxy with ⟨p₁, hp₁, p₂, hp₂, hxy⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hyz : s.WSameSide y z
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ s.WSameSide x z
  -/
  rw [wSameSide_iff_exists_left hp₂, or_iff_right hy] at hyz
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hyz : Exists fun p₂_1 => And (Membership.mem s p₂_1) (SameRay R (VSub.vsub y p …
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ s.WSameSide x z
  -/
  rcases hyz with ⟨p₃, hp₃, hyz⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub z p₃)
    ⊢ s.WSameSide x z
  -/
  refine ⟨p₁, hp₁, p₃, hp₃, hxy.trans hyz ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub z p₃)
    ⊢ Eq (VSub.vsub y p₂) 0 → Or (Eq (VSub.vsub x p₁) 0) (Eq (VSub.vsub z p₃) 0)
  -/
  refine fun h => False.elim ?_
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub z p₃)
    h : Eq (VSub.vsub y p₂) 0
    ⊢ False
  -/
  rw [vsub_eq_zero_iff_eq] at h
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub z p₃)
    h : Eq y p₂
    ⊢ False
  -/
  exact hy (h.symm ▸ hp₂)
  /-
    🎉 no goals
  -/


theorem WSameSide.trans_sSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WSameSide x y)
    (hyz : s.SSameSide y z) : s.WSameSide x z :=
  hxy.trans hyz.1 hyz.2.1


theorem WSameSide.trans_wOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WSameSide x y)
    (hyz : s.WOppSide y z) (hy : y ∉ s) : s.WOppSide x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hxy : s.WSameSide x y
    hyz : s.WOppSide y z
    hy : Not (Membership.mem s y)
    ⊢ s.WOppSide x z
  -/
  rcases hxy with ⟨p₁, hp₁, p₂, hp₂, hxy⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hyz : s.WOppSide y z
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ s.WOppSide x z
  -/
  rw [wOppSide_iff_exists_left hp₂, or_iff_right hy] at hyz
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hyz : Exists fun p₂_1 => And (Membership.mem s p₂_1) (SameRay R (VSub.vsub y p …
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    ⊢ s.WOppSide x z
  -/
  rcases hyz with ⟨p₃, hp₃, hyz⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub p₃ z)
    ⊢ s.WOppSide x z
  -/
  refine ⟨p₁, hp₁, p₃, hp₃, hxy.trans hyz ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub p₃ z)
    ⊢ Eq (VSub.vsub y p₂) 0 → Or (Eq (VSub.vsub x p₁) 0) (Eq (VSub.vsub p₃ z) 0)
  -/
  refine fun h => False.elim ?_
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub p₃ z)
    h : Eq (VSub.vsub y p₂) 0
    ⊢ False
  -/
  rw [vsub_eq_zero_iff_eq] at h
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub y p₂)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub p₃ z)
    h : Eq y p₂
    ⊢ False
  -/
  exact hy (h.symm ▸ hp₂)
  /-
    🎉 no goals
  -/


theorem WSameSide.trans_sOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WSameSide x y)
    (hyz : s.SOppSide y z) : s.WOppSide x z :=
  hxy.trans_wOppSide hyz.1 hyz.2.1


theorem SSameSide.trans_wSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SSameSide x y)
    (hyz : s.WSameSide y z) : s.WSameSide x z :=
  (hyz.symm.trans_sSameSide hxy.symm).symm


theorem SSameSide.trans {s : AffineSubspace R P} {x y z : P} (hxy : s.SSameSide x y)
    (hyz : s.SSameSide y z) : s.SSameSide x z :=
  ⟨hxy.wSameSide.trans_sSameSide hyz, hxy.2.1, hyz.2.2⟩


theorem SSameSide.trans_wOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SSameSide x y)
    (hyz : s.WOppSide y z) : s.WOppSide x z :=
  hxy.wSameSide.trans_wOppSide hyz hxy.2.2


theorem SSameSide.trans_sOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SSameSide x y)
    (hyz : s.SOppSide y z) : s.SOppSide x z :=
  ⟨hxy.trans_wOppSide hyz.1, hxy.2.1, hyz.2.2⟩


theorem WOppSide.trans_wSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WOppSide x y)
    (hyz : s.WSameSide y z) (hy : y ∉ s) : s.WOppSide x z :=
  (hyz.symm.trans_wOppSide hxy.symm hy).symm


theorem WOppSide.trans_sSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WOppSide x y)
    (hyz : s.SSameSide y z) : s.WOppSide x z :=
  hxy.trans_wSameSide hyz.1 hyz.2.1


theorem WOppSide.trans {s : AffineSubspace R P} {x y z : P} (hxy : s.WOppSide x y)
    (hyz : s.WOppSide y z) (hy : y ∉ s) : s.WSameSide x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hxy : s.WOppSide x y
    hyz : s.WOppSide y z
    hy : Not (Membership.mem s y)
    ⊢ s.WSameSide x z
  -/
  rcases hxy with ⟨p₁, hp₁, p₂, hp₂, hxy⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hyz : s.WOppSide y z
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ s.WSameSide x z
  -/
  rw [wOppSide_iff_exists_left hp₂, or_iff_right hy] at hyz
  /-
    case intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hyz : Exists fun p₂_1 => And (Membership.mem s p₂_1) (SameRay R (VSub.vsub y p …
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    ⊢ s.WSameSide x z
  -/
  rcases hyz with ⟨p₃, hp₃, hyz⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub y p₂) (VSub.vsub p₃ z)
    ⊢ s.WSameSide x z
  -/
  rw [← sameRay_neg_iff, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev] at hyz
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub p₂ y) (VSub.vsub z p₃)
    ⊢ s.WSameSide x z
  -/
  refine ⟨p₁, hp₁, p₃, hp₃, hxy.trans hyz ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub p₂ y) (VSub.vsub z p₃)
    ⊢ Eq (VSub.vsub p₂ y) 0 → Or (Eq (VSub.vsub x p₁) 0) (Eq (VSub.vsub z p₃) 0)
  -/
  refine fun h => False.elim ?_
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub p₂ y) (VSub.vsub z p₃)
    h : Eq (VSub.vsub p₂ y) 0
    ⊢ False
  -/
  rw [vsub_eq_zero_iff_eq] at h
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    hy : Not (Membership.mem s y)
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    hp₂ : Membership.mem s p₂
    hxy : SameRay R (VSub.vsub x p₁) (VSub.vsub p₂ y)
    p₃ : P
    hp₃ : Membership.mem s p₃
    hyz : SameRay R (VSub.vsub p₂ y) (VSub.vsub z p₃)
    h : Eq p₂ y
    ⊢ False
  -/
  exact hy (h ▸ hp₂)
  /-
    🎉 no goals
  -/


theorem WOppSide.trans_sOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.WOppSide x y)
    (hyz : s.SOppSide y z) : s.WSameSide x z :=
  hxy.trans hyz.1 hyz.2.1


theorem SOppSide.trans_wSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SOppSide x y)
    (hyz : s.WSameSide y z) : s.WOppSide x z :=
  (hyz.symm.trans_sOppSide hxy.symm).symm


theorem SOppSide.trans_sSameSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SOppSide x y)
    (hyz : s.SSameSide y z) : s.SOppSide x z :=
  (hyz.symm.trans_sOppSide hxy.symm).symm


theorem SOppSide.trans_wOppSide {s : AffineSubspace R P} {x y z : P} (hxy : s.SOppSide x y)
    (hyz : s.WOppSide y z) : s.WSameSide x z :=
  (hyz.symm.trans_sOppSide hxy.symm).symm


theorem SOppSide.trans {s : AffineSubspace R P} {x y z : P} (hxy : s.SOppSide x y)
    (hyz : s.SOppSide y z) : s.SSameSide x z :=
  ⟨hxy.trans_wOppSide hyz.1, hxy.2.1, hyz.2.2⟩


theorem wSameSide_and_wOppSide_iff {s : AffineSubspace R P} {x y : P} :
    s.WSameSide x y ∧ s.WOppSide x y ↔ x ∈ s ∨ y ∈ s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    ⊢ Iff (And (s.WSameSide x y) (s.WOppSide x y)) (Or (Membership.mem s x) (Membe …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      ⊢ And (s.WSameSide x y) (s.WOppSide x y) → Or (Membership.mem s x) (Membership …
    -/
  · rintro ⟨hs, ho⟩
    /-
      case mp.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      hs : s.WSameSide x y
      ho : s.WOppSide x y
      ⊢ Or (Membership.mem s x) (Membership.mem s y)
    -/
    rw [wOppSide_comm] at ho
    /-
      case mp.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      hs : s.WSameSide x y
      ho : s.WOppSide y x
      ⊢ Or (Membership.mem s x) (Membership.mem s y)
    -/
    by_contra h
    /-
      case mp.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      hs : s.WSameSide x y
      ho : s.WOppSide y x
      h : Not (Or (Membership.mem s x) (Membership.mem s y))
      ⊢ False
    -/
    rw [not_or] at h
    /-
      case mp.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      hs : s.WSameSide x y
      ho : s.WOppSide y x
      h : And (Not (Membership.mem s x)) (Not (Membership.mem s y))
      ⊢ False
    -/
    exact h.1 (wOppSide_self_iff.1 (hs.trans_wOppSide ho h.2))
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      ⊢ Or (Membership.mem s x) (Membership.mem s y) → And (s.WSameSide x y) (s.WOpp …
    -/
  · rintro (h | h)
      /-
        case mpr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y : P
        h : Membership.mem s x
        ⊢ And (s.WSameSide x y) (s.WOppSide x y)
      -/
    · exact ⟨wSameSide_of_left_mem y h, wOppSide_of_left_mem y h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y : P
        h : Membership.mem s y
        ⊢ And (s.WSameSide x y) (s.WOppSide x y)
      -/
    · exact ⟨wSameSide_of_right_mem x h, wOppSide_of_right_mem x h⟩
      /-
        🎉 no goals
      -/


theorem WSameSide.not_sOppSide {s : AffineSubspace R P} {x y : P} (h : s.WSameSide x y) :
    ¬s.SOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.WSameSide x y
    ⊢ Not (s.SOppSide x y)
  -/
  intro ho
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.WSameSide x y
    ho : s.SOppSide x y
    ⊢ False
  -/
  have hxy := wSameSide_and_wOppSide_iff.1 ⟨h, ho.1⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.WSameSide x y
    ho : s.SOppSide x y
    hxy : Or (Membership.mem s x) (Membership.mem s y)
    ⊢ False
  -/
  rcases hxy with (hx | hy)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.WSameSide x y
      ho : s.SOppSide x y
      hx : Membership.mem s x
      ⊢ False
    -/
  · exact ho.2.1 hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.WSameSide x y
      ho : s.SOppSide x y
      hy : Membership.mem s y
      ⊢ False
    -/
  · exact ho.2.2 hy
    /-
      🎉 no goals
    -/


theorem SSameSide.not_wOppSide {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) :
    ¬s.WOppSide x y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.SSameSide x y
    ⊢ Not (s.WOppSide x y)
  -/
  intro ho
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.SSameSide x y
    ho : s.WOppSide x y
    ⊢ False
  -/
  have hxy := wSameSide_and_wOppSide_iff.1 ⟨h.1, ho⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.SSameSide x y
    ho : s.WOppSide x y
    hxy : Or (Membership.mem s x) (Membership.mem s y)
    ⊢ False
  -/
  rcases hxy with (hx | hy)
    /-
      case inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.SSameSide x y
      ho : s.WOppSide x y
      hx : Membership.mem s x
      ⊢ False
    -/
  · exact h.2.1 hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.SSameSide x y
      ho : s.WOppSide x y
      hy : Membership.mem s y
      ⊢ False
    -/
  · exact h.2.2 hy
    /-
      🎉 no goals
    -/


theorem SSameSide.not_sOppSide {s : AffineSubspace R P} {x y : P} (h : s.SSameSide x y) :
    ¬s.SOppSide x y :=
  fun ho => h.not_wOppSide ho.1


theorem WOppSide.not_sSameSide {s : AffineSubspace R P} {x y : P} (h : s.WOppSide x y) :
    ¬s.SSameSide x y :=
  fun hs => hs.not_wOppSide h


theorem SOppSide.not_wSameSide {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) :
    ¬s.WSameSide x y :=
  fun hs => hs.not_sOppSide h


theorem SOppSide.not_sSameSide {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) :
    ¬s.SSameSide x y :=
  fun hs => h.not_wSameSide hs.1


theorem wOppSide_iff_exists_wbtw {s : AffineSubspace R P} {x y : P} :
    s.WOppSide x y ↔ ∃ p ∈ s, Wbtw R x p y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    ⊢ Iff (s.WOppSide x y) (Exists fun p => And (Membership.mem s p) (Wbtw R x p y))
  -/
  refine ⟨fun h => ?_, fun ⟨p, hp, h⟩ => h.wOppSide₁₃ hp⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.WOppSide x y
    ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
  -/
  rcases h with ⟨p₁, hp₁, p₂, hp₂, h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩⟩
    /-
      case intro.intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq (VSub.vsub x p₁) 0
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
    -/
  · rw [vsub_eq_zero_iff_eq] at h
    /-
      case intro.intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq x p₁
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
    -/
    rw [h]
    /-
      case intro.intro.intro.intro.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq x p₁
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R p₁ p y)
    -/
    exact ⟨p₁, hp₁, wbtw_self_left _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq (VSub.vsub p₂ y) 0
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
    -/
  · rw [vsub_eq_zero_iff_eq] at h
    /-
      case intro.intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq p₂ y
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
    -/
    rw [← h]
    /-
      case intro.intro.intro.intro.inr.inl
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      h : Eq p₂ y
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p p₂)
    -/
    exact ⟨p₂, hp₂, wbtw_self_right _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.inr.inr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y p₁ : P
      hp₁ : Membership.mem s p₁
      p₂ : P
      hp₂ : Membership.mem s p₂
      r₁ r₂ : R
      hr₁ : LT.lt 0 r₁
      hr₂ : LT.lt 0 r₂
      h : Eq (HSMul.hSMul r₁ (VSub.vsub x p₁)) (HSMul.hSMul r₂ (VSub.vsub p₂ y))
      ⊢ Exists fun p => And (Membership.mem s p) (Wbtw R x p y)
    -/
  · refine ⟨lineMap x y (r₂ / (r₁ + r₂)), ?_, ?_⟩
    · have : (r₂ / (r₁ + r₂)) • (y -ᵥ p₂ + (p₂ -ᵥ p₁) - (x -ᵥ p₁)) + (x -ᵥ p₁) =
          (r₂ / (r₁ + r₂)) • (p₂ -ᵥ p₁) := by
        rw [← neg_vsub_eq_vsub_rev p₂ y]
        linear_combination (norm := match_scalars <;> field_simp) (r₁ + r₂)⁻¹ • h
      rw [lineMap_apply, ← vsub_vadd x p₁, ← vsub_vadd y p₂, vsub_vadd_eq_vsub_sub, vadd_vsub_assoc,
        ← vadd_assoc, vadd_eq_add, this]
      /-
        case intro.intro.intro.intro.inr.inr.intro.intro.intro.intro.refine_1
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x y p₁ : P
        hp₁ : Membership.mem s p₁
        p₂ : P
        hp₂ : Membership.mem s p₂
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ (VSub.vsub x p₁)) (HSMul.hSMul r₂ (VSub.vsub p₂ y))
        this : Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂)) (HSub.hSub  …
        ⊢ Membership.mem s (HVAdd.hVAdd (HSMul.hSMul (HDiv.hDiv r₂ (HAdd.hAdd r₁ r₂))  …
      -/
      exact s.smul_vsub_vadd_mem (r₂ / (r₁ + r₂)) hp₂ hp₁ hp₁
      /-
        🎉 no goals
      -/
    · exact Set.mem_image_of_mem _
        ⟨by positivity,
          div_le_one_of_le₀ (le_add_of_nonneg_left hr₁.le) (Left.add_pos hr₁ hr₂).le⟩


theorem SOppSide.exists_sbtw {s : AffineSubspace R P} {x y : P} (h : s.SOppSide x y) :
    ∃ p ∈ s, Sbtw R x p y := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.SOppSide x y
    ⊢ Exists fun p => And (Membership.mem s p) (Sbtw R x p y)
  -/
  obtain ⟨p, hp, hw⟩ := wOppSide_iff_exists_wbtw.1 h.wOppSide
  /-
    case intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    h : s.SOppSide x y
    p : P
    hp : Membership.mem s p
    hw : Wbtw R x p y
    ⊢ Exists fun p => And (Membership.mem s p) (Sbtw R x p y)
  -/
  refine ⟨p, hp, hw, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.SOppSide x y
      p : P
      hp : Membership.mem s p
      hw : Wbtw R x p y
      ⊢ Ne p x
    -/
  · rintro rfl
    /-
      case intro.intro.refine_1
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      y p : P
      hp : Membership.mem s p
      h : s.SOppSide p y
      hw : Wbtw R p p y
      ⊢ False
    -/
    exact h.2.1 hp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x y : P
      h : s.SOppSide x y
      p : P
      hp : Membership.mem s p
      hw : Wbtw R x p y
      ⊢ Ne p y
    -/
  · rintro rfl
    /-
      case intro.intro.refine_2
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hp : Membership.mem s p
      h : s.SOppSide x p
      hw : Wbtw R x p p
      ⊢ False
    -/
    exact h.2.2 hp
    /-
      🎉 no goals
    -/


theorem _root_.Sbtw.sOppSide_of_not_mem_of_mem {s : AffineSubspace R P} {x y z : P}
    (h : Sbtw R x y z) (hx : x ∉ s) (hy : y ∈ s) : s.SOppSide x z := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    h : Sbtw R x y z
    hx : Not (Membership.mem s x)
    hy : Membership.mem s y
    ⊢ s.SOppSide x z
  -/
  refine ⟨h.wbtw.wOppSide₁₃ hy, hx, fun hz => hx ?_⟩
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y z : P
    h : Sbtw R x y z
    hx : Not (Membership.mem s x)
    hy : Membership.mem s y
    hz : Membership.mem s z
    ⊢ Membership.mem s x
  -/
  rcases h with ⟨⟨t, ⟨ht0, ht1⟩, rfl⟩, hyx, hyz⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    hx : Not (Membership.mem s x)
    hz : Membership.mem s z
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s ((AffineMap.lineMap x z) t)
    hyx : Ne ((AffineMap.lineMap x z) t) x
    hyz : Ne ((AffineMap.lineMap x z) t) z
    ⊢ Membership.mem s x
  -/
  rw [lineMap_apply] at hy
  have ht : t ≠ 1 := by
    rintro rfl
    simp [lineMap_apply] at hyz
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    hx : Not (Membership.mem s x)
    hz : Membership.mem s z
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub z x)) x)
    hyx : Ne ((AffineMap.lineMap x z) t) x
    hyz : Ne ((AffineMap.lineMap x z) t) z
    ht : Ne t 1
    ⊢ Membership.mem s x
  -/
  have hy' := vsub_mem_direction hy hz
  rw [vadd_vsub_assoc, ← neg_vsub_eq_vsub_rev z, ← neg_one_smul R (z -ᵥ x), ← add_smul,
    ← sub_eq_add_neg, s.direction.smul_mem_iff (sub_ne_zero_of_ne ht)] at hy'
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x z : P
    hx : Not (Membership.mem s x)
    hz : Membership.mem s z
    t : R
    ht0 : LE.le 0 t
    ht1 : LE.le t 1
    hy : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub z x)) x)
    hyx : Ne ((AffineMap.lineMap x z) t) x
    hyz : Ne ((AffineMap.lineMap x z) t) z
    ht : Ne t 1
    hy' : Membership.mem s.direction (VSub.vsub z x)
    ⊢ Membership.mem s x
  -/
  rwa [vadd_mem_iff_mem_of_mem_direction (Submodule.smul_mem _ _ hy')] at hy
  /-
    🎉 no goals
  -/


theorem sSameSide_smul_vsub_vadd_left {s : AffineSubspace R P} {x p₁ p₂ : P} (hx : x ∉ s)
    (hp₁ : p₁ ∈ s) (hp₂ : p₂ ∈ s) {t : R} (ht : 0 < t) : s.SSameSide (t • (x -ᵥ p₁) +ᵥ p₂) x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p₁ p₂ : P
    hx : Not (Membership.mem s x)
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LT.lt 0 t
    ⊢ s.SSameSide (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) x
  -/
  refine ⟨wSameSide_smul_vsub_vadd_left x hp₁ hp₂ ht.le, fun h => hx ?_, hx⟩
  rwa [vadd_mem_iff_mem_direction _ hp₂, s.direction.smul_mem_iff ht.ne.symm,
    vsub_right_mem_direction_iff_mem hp₁] at h


theorem sSameSide_smul_vsub_vadd_right {s : AffineSubspace R P} {x p₁ p₂ : P} (hx : x ∉ s)
    (hp₁ : p₁ ∈ s) (hp₂ : p₂ ∈ s) {t : R} (ht : 0 < t) : s.SSameSide x (t • (x -ᵥ p₁) +ᵥ p₂) :=
  (sSameSide_smul_vsub_vadd_left hx hp₁ hp₂ ht).symm


theorem sSameSide_lineMap_left {s : AffineSubspace R P} {x y : P} (hx : x ∈ s) (hy : y ∉ s) {t : R}
    (ht : 0 < t) : s.SSameSide (lineMap x y t) y :=
  sSameSide_smul_vsub_vadd_left hy hx hx ht


theorem sSameSide_lineMap_right {s : AffineSubspace R P} {x y : P} (hx : x ∈ s) (hy : y ∉ s) {t : R}
    (ht : 0 < t) : s.SSameSide y (lineMap x y t) :=
  (sSameSide_lineMap_left hx hy ht).symm


theorem sOppSide_smul_vsub_vadd_left {s : AffineSubspace R P} {x p₁ p₂ : P} (hx : x ∉ s)
    (hp₁ : p₁ ∈ s) (hp₂ : p₂ ∈ s) {t : R} (ht : t < 0) : s.SOppSide (t • (x -ᵥ p₁) +ᵥ p₂) x := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p₁ p₂ : P
    hx : Not (Membership.mem s x)
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    t : R
    ht : LT.lt t 0
    ⊢ s.SOppSide (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p₁)) p₂) x
  -/
  refine ⟨wOppSide_smul_vsub_vadd_left x hp₁ hp₂ ht.le, fun h => hx ?_, hx⟩
  rwa [vadd_mem_iff_mem_direction _ hp₂, s.direction.smul_mem_iff ht.ne,
    vsub_right_mem_direction_iff_mem hp₁] at h


theorem sOppSide_smul_vsub_vadd_right {s : AffineSubspace R P} {x p₁ p₂ : P} (hx : x ∉ s)
    (hp₁ : p₁ ∈ s) (hp₂ : p₂ ∈ s) {t : R} (ht : t < 0) : s.SOppSide x (t • (x -ᵥ p₁) +ᵥ p₂) :=
  (sOppSide_smul_vsub_vadd_left hx hp₁ hp₂ ht).symm


theorem sOppSide_lineMap_left {s : AffineSubspace R P} {x y : P} (hx : x ∈ s) (hy : y ∉ s) {t : R}
    (ht : t < 0) : s.SOppSide (lineMap x y t) y :=
  sOppSide_smul_vsub_vadd_left hy hx hx ht


theorem sOppSide_lineMap_right {s : AffineSubspace R P} {x y : P} (hx : x ∈ s) (hy : y ∉ s) {t : R}
    (ht : t < 0) : s.SOppSide y (lineMap x y t) :=
  (sOppSide_lineMap_left hx hy ht).symm


theorem setOf_wSameSide_eq_image2 {s : AffineSubspace R P} {x p : P} (hx : x ∉ s) (hp : p ∈ s) :
    { y | s.WSameSide x y } = Set.image2 (fun (t : R) q => t • (x -ᵥ p) +ᵥ q) (Set.Ici 0) s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    ⊢ Eq (setOf fun y => s.WSameSide x y) (Set.image2 (fun t q => HVAdd.hVAdd (HSM …
  -/
  ext y
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (Membership.mem (setOf fun y => s.WSameSide x y) y) (Membership.mem (Set …
  -/
  simp_rw [Set.mem_setOf, Set.mem_image2, Set.mem_Ici]
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (s.WSameSide x y) (Exists fun a => And (LE.le 0 a) (Exists fun b => And  …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ s.WSameSide x y → Exists fun a => And (LE.le 0 a) (Exists fun b => And (Memb …
    -/
  · rw [wSameSide_iff_exists_left hp, or_iff_right hx]
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun p₂ => And (Membership.mem s p₂) (SameRay R (VSub.vsub x p) (VSub …
    -/
    rintro ⟨p₂, hp₂, h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩⟩
      /-
        case h.mp.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub x p) 0
        ⊢ Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq x p
        ⊢ Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hx (h.symm ▸ hp))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub y p₂) 0
        ⊢ Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq y p₂
        ⊢ Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      refine ⟨0, le_rfl, p₂, hp₂, ?_⟩
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq y p₂
        ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul 0 (VSub.vsub x p)) p₂) y
      -/
      simp [h]
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ (VSub.vsub x p)) (HSMul.hSMul r₂ (VSub.vsub y p₂))
        ⊢ Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · refine ⟨r₁ / r₂, (div_pos hr₁ hr₂).le, p₂, hp₂, ?_⟩
      rw [div_eq_inv_mul, ← smul_smul, h, smul_smul, inv_mul_cancel₀ hr₂.ne.symm, one_smul,
        vsub_vadd]
    /-
      case h.mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun a => And (LE.le 0 a) (Exists fun b => And (Membership.mem (↑s) b …
    -/
  · rintro ⟨t, ht, p', hp', rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      t : R
      ht : LE.le 0 t
      p' : P
      hp' : Membership.mem (↑s) p'
      ⊢ s.WSameSide x (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p)) p')
    -/
    exact wSameSide_smul_vsub_vadd_right x hp hp' ht
    /-
      🎉 no goals
    -/


theorem setOf_sSameSide_eq_image2 {s : AffineSubspace R P} {x p : P} (hx : x ∉ s) (hp : p ∈ s) :
    { y | s.SSameSide x y } = Set.image2 (fun (t : R) q => t • (x -ᵥ p) +ᵥ q) (Set.Ioi 0) s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    ⊢ Eq (setOf fun y => s.SSameSide x y) (Set.image2 (fun t q => HVAdd.hVAdd (HSM …
  -/
  ext y
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (Membership.mem (setOf fun y => s.SSameSide x y) y) (Membership.mem (Set …
  -/
  simp_rw [Set.mem_setOf, Set.mem_image2, Set.mem_Ioi]
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (s.SSameSide x y) (Exists fun a => And (LT.lt 0 a) (Exists fun b => And  …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ s.SSameSide x y → Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Memb …
    -/
  · rw [sSameSide_iff_exists_left hp]
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ And (Not (Membership.mem s x)) (And (Not (Membership.mem s y)) (Exists fun p …
    -/
    rintro ⟨-, hy, p₂, hp₂, h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩⟩
      /-
        case h.mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub x p) 0
        ⊢ Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq x p
        ⊢ Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hx (h.symm ▸ hp))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub y p₂) 0
        ⊢ Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq y p₂
        ⊢ Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hy (h.symm ▸ hp₂))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ (VSub.vsub x p)) (HSMul.hSMul r₂ (VSub.vsub y p₂))
        ⊢ Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · refine ⟨r₁ / r₂, div_pos hr₁ hr₂, p₂, hp₂, ?_⟩
      rw [div_eq_inv_mul, ← smul_smul, h, smul_smul, inv_mul_cancel₀ hr₂.ne.symm, one_smul,
        vsub_vadd]
    /-
      case h.mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun a => And (LT.lt 0 a) (Exists fun b => And (Membership.mem (↑s) b …
    -/
  · rintro ⟨t, ht, p', hp', rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      t : R
      ht : LT.lt 0 t
      p' : P
      hp' : Membership.mem (↑s) p'
      ⊢ s.SSameSide x (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p)) p')
    -/
    exact sSameSide_smul_vsub_vadd_right hx hp hp' ht
    /-
      🎉 no goals
    -/


theorem setOf_wOppSide_eq_image2 {s : AffineSubspace R P} {x p : P} (hx : x ∉ s) (hp : p ∈ s) :
    { y | s.WOppSide x y } = Set.image2 (fun (t : R) q => t • (x -ᵥ p) +ᵥ q) (Set.Iic 0) s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    ⊢ Eq (setOf fun y => s.WOppSide x y) (Set.image2 (fun t q => HVAdd.hVAdd (HSMu …
  -/
  ext y
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (Membership.mem (setOf fun y => s.WOppSide x y) y) (Membership.mem (Set. …
  -/
  simp_rw [Set.mem_setOf, Set.mem_image2, Set.mem_Iic]
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (s.WOppSide x y) (Exists fun a => And (LE.le a 0) (Exists fun b => And ( …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ s.WOppSide x y → Exists fun a => And (LE.le a 0) (Exists fun b => And (Membe …
    -/
  · rw [wOppSide_iff_exists_left hp, or_iff_right hx]
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun p₂ => And (Membership.mem s p₂) (SameRay R (VSub.vsub x p) (VSub …
    -/
    rintro ⟨p₂, hp₂, h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩⟩
      /-
        case h.mp.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub x p) 0
        ⊢ Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq x p
        ⊢ Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hx (h.symm ▸ hp))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub p₂ y) 0
        ⊢ Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq p₂ y
        ⊢ Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      refine ⟨0, le_rfl, p₂, hp₂, ?_⟩
      /-
        case h.mp.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq p₂ y
        ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul 0 (VSub.vsub x p)) p₂) y
      -/
      simp [h]
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y p₂ : P
        hp₂ : Membership.mem s p₂
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ (VSub.vsub x p)) (HSMul.hSMul r₂ (VSub.vsub p₂ y))
        ⊢ Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · refine ⟨-r₁ / r₂, (div_neg_of_neg_of_pos (Left.neg_neg_iff.2 hr₁) hr₂).le, p₂, hp₂, ?_⟩
      rw [div_eq_inv_mul, ← smul_smul, neg_smul, h, smul_neg, smul_smul,
        inv_mul_cancel₀ hr₂.ne.symm, one_smul, neg_vsub_eq_vsub_rev, vsub_vadd]
    /-
      case h.mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun a => And (LE.le a 0) (Exists fun b => And (Membership.mem (↑s) b …
    -/
  · rintro ⟨t, ht, p', hp', rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      t : R
      ht : LE.le t 0
      p' : P
      hp' : Membership.mem (↑s) p'
      ⊢ s.WOppSide x (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p)) p')
    -/
    exact wOppSide_smul_vsub_vadd_right x hp hp' ht
    /-
      🎉 no goals
    -/


theorem setOf_sOppSide_eq_image2 {s : AffineSubspace R P} {x p : P} (hx : x ∉ s) (hp : p ∈ s) :
    { y | s.SOppSide x y } = Set.image2 (fun (t : R) q => t • (x -ᵥ p) +ᵥ q) (Set.Iio 0) s := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    ⊢ Eq (setOf fun y => s.SOppSide x y) (Set.image2 (fun t q => HVAdd.hVAdd (HSMu …
  -/
  ext y
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (Membership.mem (setOf fun y => s.SOppSide x y) y) (Membership.mem (Set. …
  -/
  simp_rw [Set.mem_setOf, Set.mem_image2, Set.mem_Iio]
  /-
    case h
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x p : P
    hx : Not (Membership.mem s x)
    hp : Membership.mem s p
    y : P
    ⊢ Iff (s.SOppSide x y) (Exists fun a => And (LT.lt a 0) (Exists fun b => And ( …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ s.SOppSide x y → Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membe …
    -/
  · rw [sOppSide_iff_exists_left hp]
    /-
      case h.mp
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ And (Not (Membership.mem s x)) (And (Not (Membership.mem s y)) (Exists fun p …
    -/
    rintro ⟨-, hy, p₂, hp₂, h | h | ⟨r₁, r₂, hr₁, hr₂, h⟩⟩
      /-
        case h.mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub x p) 0
        ⊢ Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.intro.intro.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq x p
        ⊢ Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hx (h.symm ▸ hp))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq (VSub.vsub p₂ y) 0
        ⊢ Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · rw [vsub_eq_zero_iff_eq] at h
      /-
        case h.mp.intro.intro.intro.intro.inr.inl
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        h : Eq p₂ y
        ⊢ Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
      exact False.elim (hy (h ▸ hp₂))
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.intro.inr.inr.intro.intro.intro.intro
        R : Type u_1
        V : Type u_2
        P : Type u_4
        inst✝³ : LinearOrderedField R
        inst✝² : AddCommGroup V
        inst✝¹ : Module R V
        inst✝ : AddTorsor V P
        s : AffineSubspace R P
        x p : P
        hx : Not (Membership.mem s x)
        hp : Membership.mem s p
        y : P
        hy : Not (Membership.mem s y)
        p₂ : P
        hp₂ : Membership.mem s p₂
        r₁ r₂ : R
        hr₁ : LT.lt 0 r₁
        hr₂ : LT.lt 0 r₂
        h : Eq (HSMul.hSMul r₁ (VSub.vsub x p)) (HSMul.hSMul r₂ (VSub.vsub p₂ y))
        ⊢ Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b) …
      -/
    · refine ⟨-r₁ / r₂, div_neg_of_neg_of_pos (Left.neg_neg_iff.2 hr₁) hr₂, p₂, hp₂, ?_⟩
      rw [div_eq_inv_mul, ← smul_smul, neg_smul, h, smul_neg, smul_smul,
        inv_mul_cancel₀ hr₂.ne.symm, one_smul, neg_vsub_eq_vsub_rev, vsub_vadd]
    /-
      case h.mpr
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      y : P
      ⊢ (Exists fun a => And (LT.lt a 0) (Exists fun b => And (Membership.mem (↑s) b …
    -/
  · rintro ⟨t, ht, p', hp', rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro
      R : Type u_1
      V : Type u_2
      P : Type u_4
      inst✝³ : LinearOrderedField R
      inst✝² : AddCommGroup V
      inst✝¹ : Module R V
      inst✝ : AddTorsor V P
      s : AffineSubspace R P
      x p : P
      hx : Not (Membership.mem s x)
      hp : Membership.mem s p
      t : R
      ht : LT.lt t 0
      p' : P
      hp' : Membership.mem (↑s) p'
      ⊢ s.SOppSide x (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub x p)) p')
    -/
    exact sOppSide_smul_vsub_vadd_right hx hp hp' ht
    /-
      🎉 no goals
    -/


theorem wOppSide_pointReflection {s : AffineSubspace R P} {x : P} (y : P) (hx : x ∈ s) :
    s.WOppSide y (pointReflection R x y) :=
  (wbtw_pointReflection R _ _).wOppSide₁₃ hx


theorem sOppSide_pointReflection {s : AffineSubspace R P} {x y : P} (hx : x ∈ s) (hy : y ∉ s) :
    s.SOppSide y (pointReflection R x y) := by
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    hy : Not (Membership.mem s y)
    ⊢ s.SOppSide y ((AffineEquiv.pointReflection R x) y)
  -/
  refine (sbtw_pointReflection_of_ne R fun h => hy ?_).sOppSide_of_not_mem_of_mem hy hx
  /-
    R : Type u_1
    V : Type u_2
    P : Type u_4
    inst✝³ : LinearOrderedField R
    inst✝² : AddCommGroup V
    inst✝¹ : Module R V
    inst✝ : AddTorsor V P
    s : AffineSubspace R P
    x y : P
    hx : Membership.mem s x
    hy : Not (Membership.mem s y)
    h : Eq x y
    ⊢ Membership.mem s y
  -/
  rwa [← h]
  /-
    🎉 no goals
  -/


theorem isConnected_setOf_wSameSide {s : AffineSubspace ℝ P} (x : P) (h : (s : Set P).Nonempty) :
    IsConnected { y | s.WSameSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    h : (↑s).Nonempty
    ⊢ IsConnected (setOf fun y => s.WSameSide x y)
  -/
  obtain ⟨p, hp⟩ := h
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x p : P
    hp : Membership.mem (↑s) p
    ⊢ IsConnected (setOf fun y => s.WSameSide x y)
  -/
  haveI : Nonempty s := ⟨⟨p, hp⟩⟩
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ IsConnected (setOf fun y => s.WSameSide x y)
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      ⊢ IsConnected (setOf fun y => s.WSameSide x y)
    -/
  · simp only [wSameSide_of_left_mem, hx]
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      ⊢ IsConnected (setOf fun y => True)
    -/
    have := AddTorsor.connectedSpace V P
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this✝ : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      this : ConnectedSpace P
      ⊢ IsConnected (setOf fun y => True)
    -/
    exact isConnected_univ
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Not (Membership.mem s x)
      ⊢ IsConnected (setOf fun y => s.WSameSide x y)
    -/
  · rw [setOf_wSameSide_eq_image2 hx hp, ← Set.image_prod]
    refine (isConnected_Ici.prod (isConnected_iff_connectedSpace.2 ?_)).image _
      ((continuous_fst.smul continuous_const).vadd continuous_snd).continuousOn
    /-
      case neg
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Not (Membership.mem s x)
      ⊢ ConnectedSpace ↑↑s
    -/
    convert AddTorsor.connectedSpace s.direction s
    /-
      🎉 no goals
    -/


theorem isPreconnected_setOf_wSameSide (s : AffineSubspace ℝ P) (x : P) :
    IsPreconnected { y | s.WSameSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    ⊢ IsPreconnected (setOf fun y => s.WSameSide x y)
  -/
  rcases Set.eq_empty_or_nonempty (s : Set P) with (h | h)
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ IsPreconnected (setOf fun y => s.WSameSide x y)
    -/
  · rw [coe_eq_bot_iff] at h
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => s.WSameSide x y)
    -/
    simp only [h, not_wSameSide_bot]
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => False)
    -/
    exact isPreconnected_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : (↑s).Nonempty
      ⊢ IsPreconnected (setOf fun y => s.WSameSide x y)
    -/
  · exact (isConnected_setOf_wSameSide x h).isPreconnected
    /-
      🎉 no goals
    -/


theorem isConnected_setOf_sSameSide {s : AffineSubspace ℝ P} {x : P} (hx : x ∉ s)
    (h : (s : Set P).Nonempty) : IsConnected { y | s.SSameSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    h : (↑s).Nonempty
    ⊢ IsConnected (setOf fun y => s.SSameSide x y)
  -/
  obtain ⟨p, hp⟩ := h
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    ⊢ IsConnected (setOf fun y => s.SSameSide x y)
  -/
  haveI : Nonempty s := ⟨⟨p, hp⟩⟩
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ IsConnected (setOf fun y => s.SSameSide x y)
  -/
  rw [setOf_sSameSide_eq_image2 hx hp, ← Set.image_prod]
  refine (isConnected_Ioi.prod (isConnected_iff_connectedSpace.2 ?_)).image _
    ((continuous_fst.smul continuous_const).vadd continuous_snd).continuousOn
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ ConnectedSpace ↑↑s
  -/
  convert AddTorsor.connectedSpace s.direction s
  /-
    🎉 no goals
  -/


theorem isPreconnected_setOf_sSameSide (s : AffineSubspace ℝ P) (x : P) :
    IsPreconnected { y | s.SSameSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
  -/
  rcases Set.eq_empty_or_nonempty (s : Set P) with (h | h)
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
    -/
  · rw [coe_eq_bot_iff] at h
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
    -/
    simp only [h, not_sSameSide_bot]
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => False)
    -/
    exact isPreconnected_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : (↑s).Nonempty
      ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
    -/
  · by_cases hx : x ∈ s
      /-
        case pos
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Membership.mem s x
        ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
      -/
    · simp only [hx, SSameSide, not_true, false_and, and_false]
      /-
        case pos
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Membership.mem s x
        ⊢ IsPreconnected (setOf fun y => False)
      -/
      exact isPreconnected_empty
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Not (Membership.mem s x)
        ⊢ IsPreconnected (setOf fun y => s.SSameSide x y)
      -/
    · exact (isConnected_setOf_sSameSide hx h).isPreconnected
      /-
        🎉 no goals
      -/


theorem isConnected_setOf_wOppSide {s : AffineSubspace ℝ P} (x : P) (h : (s : Set P).Nonempty) :
    IsConnected { y | s.WOppSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    h : (↑s).Nonempty
    ⊢ IsConnected (setOf fun y => s.WOppSide x y)
  -/
  obtain ⟨p, hp⟩ := h
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x p : P
    hp : Membership.mem (↑s) p
    ⊢ IsConnected (setOf fun y => s.WOppSide x y)
  -/
  haveI : Nonempty s := ⟨⟨p, hp⟩⟩
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ IsConnected (setOf fun y => s.WOppSide x y)
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      ⊢ IsConnected (setOf fun y => s.WOppSide x y)
    -/
  · simp only [wOppSide_of_left_mem, hx]
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      ⊢ IsConnected (setOf fun y => True)
    -/
    have := AddTorsor.connectedSpace V P
    /-
      case pos
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this✝ : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Membership.mem s x
      this : ConnectedSpace P
      ⊢ IsConnected (setOf fun y => True)
    -/
    exact isConnected_univ
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Not (Membership.mem s x)
      ⊢ IsConnected (setOf fun y => s.WOppSide x y)
    -/
  · rw [setOf_wOppSide_eq_image2 hx hp, ← Set.image_prod]
    refine (isConnected_Iic.prod (isConnected_iff_connectedSpace.2 ?_)).image _
      ((continuous_fst.smul continuous_const).vadd continuous_snd).continuousOn
    /-
      case neg
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x p : P
      hp : Membership.mem (↑s) p
      this : Nonempty (Subtype fun x => Membership.mem s x)
      hx : Not (Membership.mem s x)
      ⊢ ConnectedSpace ↑↑s
    -/
    convert AddTorsor.connectedSpace s.direction s
    /-
      🎉 no goals
    -/


theorem isPreconnected_setOf_wOppSide (s : AffineSubspace ℝ P) (x : P) :
    IsPreconnected { y | s.WOppSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    ⊢ IsPreconnected (setOf fun y => s.WOppSide x y)
  -/
  rcases Set.eq_empty_or_nonempty (s : Set P) with (h | h)
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ IsPreconnected (setOf fun y => s.WOppSide x y)
    -/
  · rw [coe_eq_bot_iff] at h
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => s.WOppSide x y)
    -/
    simp only [h, not_wOppSide_bot]
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => False)
    -/
    exact isPreconnected_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : (↑s).Nonempty
      ⊢ IsPreconnected (setOf fun y => s.WOppSide x y)
    -/
  · exact (isConnected_setOf_wOppSide x h).isPreconnected
    /-
      🎉 no goals
    -/


theorem isConnected_setOf_sOppSide {s : AffineSubspace ℝ P} {x : P} (hx : x ∉ s)
    (h : (s : Set P).Nonempty) : IsConnected { y | s.SOppSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    h : (↑s).Nonempty
    ⊢ IsConnected (setOf fun y => s.SOppSide x y)
  -/
  obtain ⟨p, hp⟩ := h
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    ⊢ IsConnected (setOf fun y => s.SOppSide x y)
  -/
  haveI : Nonempty s := ⟨⟨p, hp⟩⟩
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ IsConnected (setOf fun y => s.SOppSide x y)
  -/
  rw [setOf_sOppSide_eq_image2 hx hp, ← Set.image_prod]
  refine (isConnected_Iio.prod (isConnected_iff_connectedSpace.2 ?_)).image _
    ((continuous_fst.smul continuous_const).vadd continuous_snd).continuousOn
  /-
    case intro
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    hx : Not (Membership.mem s x)
    p : P
    hp : Membership.mem (↑s) p
    this : Nonempty (Subtype fun x => Membership.mem s x)
    ⊢ ConnectedSpace ↑↑s
  -/
  convert AddTorsor.connectedSpace s.direction s
  /-
    🎉 no goals
  -/


theorem isPreconnected_setOf_sOppSide (s : AffineSubspace ℝ P) (x : P) :
    IsPreconnected { y | s.SOppSide x y } := by
  /-
    V : Type u_2
    P : Type u_4
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : AffineSubspace Real P
    x : P
    ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
  -/
  rcases Set.eq_empty_or_nonempty (s : Set P) with (h | h)
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq (↑s) EmptyCollection.emptyCollection
      ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
    -/
  · rw [coe_eq_bot_iff] at h
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
    -/
    simp only [h, not_sOppSide_bot]
    /-
      case inl
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : Eq s Bot.bot
      ⊢ IsPreconnected (setOf fun y => False)
    -/
    exact isPreconnected_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_2
      P : Type u_4
      inst✝³ : SeminormedAddCommGroup V
      inst✝² : NormedSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      s : AffineSubspace Real P
      x : P
      h : (↑s).Nonempty
      ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
    -/
  · by_cases hx : x ∈ s
      /-
        case pos
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Membership.mem s x
        ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
      -/
    · simp only [hx, SOppSide, not_true, false_and, and_false]
      /-
        case pos
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Membership.mem s x
        ⊢ IsPreconnected (setOf fun y => False)
      -/
      exact isPreconnected_empty
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_2
        P : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : NormedSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        x : P
        h : (↑s).Nonempty
        hx : Not (Membership.mem s x)
        ⊢ IsPreconnected (setOf fun y => s.SOppSide x y)
      -/
    · exact (isConnected_setOf_sOppSide hx h).isPreconnected
      /-
        🎉 no goals
      -/


