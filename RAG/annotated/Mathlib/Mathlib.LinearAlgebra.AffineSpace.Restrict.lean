theorem AffineSubspace.nonempty_map {E : AffineSubspace k P₁} [Ene : Nonempty E] {φ : P₁ →ᵃ[k] P₂} :
    Nonempty (E.map φ) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : AddCommGroup V₂
    inst✝³ : Module k V₁
    inst✝² : Module k V₂
    inst✝¹ : AddTorsor V₁ P₁
    inst✝ : AddTorsor V₂ P₂
    E : AffineSubspace k P₁
    Ene : Nonempty (Subtype fun x => Membership.mem E x)
    φ : AffineMap k P₁ P₂
    ⊢ Nonempty (Subtype fun x => Membership.mem (AffineSubspace.map φ E) x)
  -/
  obtain ⟨x, hx⟩ := id Ene
  /-
    case intro.mk
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : AddCommGroup V₂
    inst✝³ : Module k V₁
    inst✝² : Module k V₂
    inst✝¹ : AddTorsor V₁ P₁
    inst✝ : AddTorsor V₂ P₂
    E : AffineSubspace k P₁
    Ene : Nonempty (Subtype fun x => Membership.mem E x)
    φ : AffineMap k P₁ P₂
    x : P₁
    hx : Membership.mem E x
    ⊢ Nonempty (Subtype fun x => Membership.mem (AffineSubspace.map φ E) x)
  -/
  exact ⟨⟨φ x, AffineSubspace.mem_map.mpr ⟨x, hx, rfl⟩⟩⟩
  /-
    🎉 no goals
  -/

-- Porting note: removed "local nolint fails_quickly" attribute

/-- Restrict domain and codomain of an affine map to the given subspaces. -/
def AffineMap.restrict (φ : P₁ →ᵃ[k] P₂) {E : AffineSubspace k P₁} {F : AffineSubspace k P₂}
    [Nonempty E] [Nonempty F] (hEF : E.map φ ≤ F) : E →ᵃ[k] F := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    hEF : LE.le (AffineSubspace.map φ E) F
    ⊢ AffineMap k (Subtype fun x => Membership.mem E x) (Subtype fun x => Membersh …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case refine_1
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      ⊢ (Subtype fun x => Membership.mem E x) → Subtype fun x => Membership.mem F x
    -/
  · exact fun x => ⟨φ x, hEF <| AffineSubspace.mem_map.mpr ⟨x, x.property, rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      ⊢ LinearMap (RingHom.id k) (Subtype fun x => Membership.mem E.direction x) (Su …
    -/
  · refine φ.linear.restrict (?_ : E.direction ≤ F.direction.comap φ.linear)
    /-
      case refine_2
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      ⊢ LE.le E.direction (Submodule.comap φ.linear F.direction)
    -/
    rw [← Submodule.map_le_iff_le_comap, ← AffineSubspace.map_direction]
    /-
      case refine_2
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      ⊢ LE.le (AffineSubspace.map φ E).direction F.direction
    -/
    exact AffineSubspace.direction_le hEF
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      ⊢ ∀ (p : Subtype fun x => Membership.mem E x) (v : Subtype fun x => Membership …
    -/
  · intro p v
    /-
      case refine_3
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      p : Subtype fun x => Membership.mem E x
      v : Subtype fun x => Membership.mem E.direction x
      ⊢ Eq ⟨φ ↑(HVAdd.hVAdd v p), ⋯⟩ (HVAdd.hVAdd ((φ.linear.restrict ⋯) v) ⟨φ ↑p, ⋯⟩)
    -/
    simp only [Subtype.ext_iff, Subtype.coe_mk, AffineSubspace.coe_vadd]
    /-
      case refine_3
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁸ : Ring k
      inst✝⁷ : AddCommGroup V₁
      inst✝⁶ : AddCommGroup V₂
      inst✝⁵ : Module k V₁
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddTorsor V₂ P₂
      φ : AffineMap k P₁ P₂
      E : AffineSubspace k P₁
      F : AffineSubspace k P₂
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
      inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
      hEF : LE.le (AffineSubspace.map φ E) F
      p : Subtype fun x => Membership.mem E x
      v : Subtype fun x => Membership.mem E.direction x
      ⊢ Eq (φ (HVAdd.hVAdd ↑v ↑p)) (HVAdd.hVAdd (↑((φ.linear.restrict ⋯) v)) (φ ↑p))
    -/
    apply AffineMap.map_vadd
    /-
      🎉 no goals
    -/


theorem AffineMap.restrict.coe_apply (φ : P₁ →ᵃ[k] P₂) {E : AffineSubspace k P₁}
    {F : AffineSubspace k P₂} [Nonempty E] [Nonempty F] (hEF : E.map φ ≤ F) (x : E) :
    ↑(φ.restrict hEF x) = φ x :=
  rfl


theorem AffineMap.restrict.linear_aux {φ : P₁ →ᵃ[k] P₂} {E : AffineSubspace k P₁}
    {F : AffineSubspace k P₂} (hEF : E.map φ ≤ F) : E.direction ≤ F.direction.comap φ.linear := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : AddCommGroup V₂
    inst✝³ : Module k V₁
    inst✝² : Module k V₂
    inst✝¹ : AddTorsor V₁ P₁
    inst✝ : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    hEF : LE.le (AffineSubspace.map φ E) F
    ⊢ LE.le E.direction (Submodule.comap φ.linear F.direction)
  -/
  rw [← Submodule.map_le_iff_le_comap, ← AffineSubspace.map_direction]
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : AddCommGroup V₂
    inst✝³ : Module k V₁
    inst✝² : Module k V₂
    inst✝¹ : AddTorsor V₁ P₁
    inst✝ : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    hEF : LE.le (AffineSubspace.map φ E) F
    ⊢ LE.le (AffineSubspace.map φ E).direction F.direction
  -/
  exact AffineSubspace.direction_le hEF
  /-
    🎉 no goals
  -/


theorem AffineMap.restrict.linear (φ : P₁ →ᵃ[k] P₂) {E : AffineSubspace k P₁}
    {F : AffineSubspace k P₂} [Nonempty E] [Nonempty F] (hEF : E.map φ ≤ F) :
    (φ.restrict hEF).linear = φ.linear.restrict (AffineMap.restrict.linear_aux hEF) :=
  rfl


theorem AffineMap.restrict.injective {φ : P₁ →ᵃ[k] P₂} (hφ : Function.Injective φ)
    {E : AffineSubspace k P₁} {F : AffineSubspace k P₂} [Nonempty E] [Nonempty F]
    (hEF : E.map φ ≤ F) : Function.Injective (AffineMap.restrict φ hEF) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    hφ : Function.Injective ⇑φ
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    hEF : LE.le (AffineSubspace.map φ E) F
    ⊢ Function.Injective ⇑(φ.restrict hEF)
  -/
  intro x y h
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    hφ : Function.Injective ⇑φ
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    hEF : LE.le (AffineSubspace.map φ E) F
    x y : Subtype fun x => Membership.mem E x
    h : Eq ((φ.restrict hEF) x) ((φ.restrict hEF) y)
    ⊢ Eq x y
  -/
  simp only [Subtype.ext_iff, Subtype.coe_mk, AffineMap.restrict.coe_apply] at h ⊢
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    hφ : Function.Injective ⇑φ
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    hEF : LE.le (AffineSubspace.map φ E) F
    x y : Subtype fun x => Membership.mem E x
    h : Eq (φ ↑x) (φ ↑y)
    ⊢ Eq ↑x ↑y
  -/
  exact hφ h
  /-
    🎉 no goals
  -/


theorem AffineMap.restrict.surjective (φ : P₁ →ᵃ[k] P₂) {E : AffineSubspace k P₁}
    {F : AffineSubspace k P₂} [Nonempty E] [Nonempty F] (h : E.map φ = F) :
    Function.Surjective (AffineMap.restrict φ (le_of_eq h)) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    h : Eq (AffineSubspace.map φ E) F
    ⊢ Function.Surjective ⇑(φ.restrict ⋯)
  -/
  rintro ⟨x, hx : x ∈ F⟩
  /-
    case mk
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    h : Eq (AffineSubspace.map φ E) F
    x : P₂
    hx : Membership.mem F x
    ⊢ Exists fun a => Eq ((φ.restrict ⋯) a) ⟨x, hx⟩
  -/
  rw [← h, AffineSubspace.mem_map] at hx
  /-
    case mk
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    h : Eq (AffineSubspace.map φ E) F
    x : P₂
    hx✝ : Membership.mem F x
    hx : Exists fun y => And (Membership.mem E y) (Eq (φ y) x)
    ⊢ Exists fun a => Eq ((φ.restrict ⋯) a) ⟨x, hx✝⟩
  -/
  obtain ⟨y, hy, rfl⟩ := hx
  /-
    case mk.intro.intro
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V₁
    inst✝⁶ : AddCommGroup V₂
    inst✝⁵ : Module k V₁
    inst✝⁴ : Module k V₂
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddTorsor V₂ P₂
    φ : AffineMap k P₁ P₂
    E : AffineSubspace k P₁
    F : AffineSubspace k P₂
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem E x)
    inst✝ : Nonempty (Subtype fun x => Membership.mem F x)
    h : Eq (AffineSubspace.map φ E) F
    y : P₁
    hy : Membership.mem E y
    hx : Membership.mem F (φ y)
    ⊢ Exists fun a => Eq ((φ.restrict ⋯) a) ⟨φ y, hx⟩
  -/
  exact ⟨⟨y, hy⟩, rfl⟩
  /-
    🎉 no goals
  -/


theorem AffineMap.restrict.bijective {E : AffineSubspace k P₁} [Nonempty E] {φ : P₁ →ᵃ[k] P₂}
    (hφ : Function.Injective φ) : Function.Bijective (φ.restrict (le_refl (E.map φ))) :=
  ⟨AffineMap.restrict.injective hφ _, AffineMap.restrict.surjective _ rfl⟩

