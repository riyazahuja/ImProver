/-- Inducts on the base `b` expansion of an ordinal. -/
@[elab_as_elim]
noncomputable def CNFRec (b : Ordinal) {C : Ordinal → Sort*} (H0 : C 0)
    (H : ∀ o, o ≠ 0 → C (o % b ^ log b o) → C o) (o : Ordinal) : C o :=
  if h : o = 0 then h ▸ H0 else H o h (CNFRec b H0 H (o % b ^ log b o))
termination_by o
/-
  b o : Ordinal.{u_2}
  h : Not (Eq o 0)
  ⊢ LT.lt (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) o
-/
decreasing_by exact mod_opow_log_lt_self b h
/-
  🎉 no goals
-/


@[simp]
theorem CNFRec_zero {C : Ordinal → Sort*} (b : Ordinal) (H0 : C 0)
    (H : ∀ o, o ≠ 0 → C (o % b ^ log b o) → C o) : CNFRec b H0 H 0 = H0 := by
  /-
    C : Ordinal.{u_2} → Sort u_1
    b : Ordinal.{u_2}
    H0 : C 0
    H : (o : Ordinal.{u_2}) → Ne o 0 → C (HMod.hMod o (HPow.hPow b (Ordinal.log b  …
    ⊢ Eq (b.CNFRec H0 H 0) H0
  -/
  rw [CNFRec, dif_pos rfl]
  /-
    🎉 no goals
  -/


theorem CNFRec_pos (b : Ordinal) {o : Ordinal} {C : Ordinal → Sort*} (ho : o ≠ 0) (H0 : C 0)
    (H : ∀ o, o ≠ 0 → C (o % b ^ log b o) → C o) :
    CNFRec b H0 H o = H o ho (@CNFRec b C H0 H _) := by
  /-
    b o : Ordinal.{u_2}
    C : Ordinal.{u_2} → Sort u_1
    ho : Ne o 0
    H0 : C 0
    H : (o : Ordinal.{u_2}) → Ne o 0 → C (HMod.hMod o (HPow.hPow b (Ordinal.log b  …
    ⊢ Eq (b.CNFRec H0 H o) (H o ho (b.CNFRec H0 H (HMod.hMod o (HPow.hPow b (Ordin …
  -/
  rw [CNFRec, dif_neg]
  /-
    🎉 no goals
  -/


/-- The Cantor normal form of an ordinal `o` is the list of coefficients and exponents in the
base-`b` expansion of `o`.

We special-case `CNF 0 o = CNF 1 o = [(0, o)]` for `o ≠ 0`.

`CNF b (b ^ u₁ * v₁ + b ^ u₂ * v₂) = [(u₁, v₁), (u₂, v₂)]` -/
@[pp_nodot]
def CNF (b o : Ordinal) : List (Ordinal × Ordinal) :=
  CNFRec b [] (fun o _ IH ↦ (log b o, o / b ^ log b o)::IH) o


@[simp]
theorem CNF_zero (b : Ordinal) : CNF b 0 = [] :=
  CNFRec_zero b _ _


/-- Recursive definition for the Cantor normal form. -/
theorem CNF_ne_zero {b o : Ordinal} (ho : o ≠ 0) :
    CNF b o = (log b o, o / b ^ log b o)::CNF b (o % b ^ log b o) :=
  CNFRec_pos b ho _ _


                                                                       /-
                                                                         o : Ordinal.{u_1}
                                                                         ho : Ne o 0
                                                                         ⊢ Eq (Ordinal.CNF 0 o) (List.cons { fst := 0, snd := o } List.nil)
                                                                       -/
theorem zero_CNF {o : Ordinal} (ho : o ≠ 0) : CNF 0 o = [(0, o)] := by simp [CNF_ne_zero ho]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                      /-
                                                                        o : Ordinal.{u_1}
                                                                        ho : Ne o 0
                                                                        ⊢ Eq (Ordinal.CNF 1 o) (List.cons { fst := 0, snd := o } List.nil)
                                                                      -/
theorem one_CNF {o : Ordinal} (ho : o ≠ 0) : CNF 1 o = [(0, o)] := by simp [CNF_ne_zero ho]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem CNF_of_le_one {b o : Ordinal} (hb : b ≤ 1) (ho : o ≠ 0) : CNF b o = [(0, o)] := by
  /-
    b o : Ordinal.{u_1}
    hb : LE.le b 1
    ho : Ne o 0
    ⊢ Eq (Ordinal.CNF b o) (List.cons { fst := 0, snd := o } List.nil)
  -/
  rcases le_one_iff.1 hb with (rfl | rfl)
  /-
    case inl
    o : Ordinal.{u_1}
    ho : Ne o 0
    hb : LE.le 0 1
    ⊢ Eq (Ordinal.CNF 0 o) (List.cons { fst := 0, snd := o } List.nil)
  -/
  exacts [zero_CNF ho, one_CNF ho]
  /-
    🎉 no goals
  -/


theorem CNF_of_lt {b o : Ordinal} (ho : o ≠ 0) (hb : o < b) : CNF b o = [(0, o)] := by
  /-
    b o : Ordinal.{u_1}
    ho : Ne o 0
    hb : LT.lt o b
    ⊢ Eq (Ordinal.CNF b o) (List.cons { fst := 0, snd := o } List.nil)
  -/
  rw [CNF_ne_zero ho, log_eq_zero hb, opow_zero, div_one, mod_one, CNF_zero]
  /-
    🎉 no goals
  -/


/-- Evaluating the Cantor normal form of an ordinal returns the ordinal. -/
theorem CNF_foldr (b o : Ordinal) : (CNF b o).foldr (fun p r ↦ b ^ p.1 * p.2 + r) 0 = o := by
  /-
    b o : Ordinal.{u_1}
    ⊢ Eq (List.foldr (fun p r => HAdd.hAdd (HMul.hMul (HPow.hPow b p.1) p.2) r) 0  …
  -/
  refine CNFRec b ?_ ?_ o
    /-
      case refine_1
      b o : Ordinal.{u_1}
      ⊢ Eq (List.foldr (fun p r => HAdd.hAdd (HMul.hMul (HPow.hPow b p.1) p.2) r) 0  …
    -/
  · rw [CNF_zero, foldr_nil]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      b o : Ordinal.{u_1}
      ⊢ ∀ (o : Ordinal.{u_1}), Ne o 0 → Eq (List.foldr (fun p r => HAdd.hAdd (HMul.h …
    -/
  · intro o ho IH
    /-
      case refine_2
      b o✝ o : Ordinal.{u_1}
      ho : Ne o 0
      IH : Eq (List.foldr (fun p r => HAdd.hAdd (HMul.hMul (HPow.hPow b p.1) p.2) r) …
      ⊢ Eq (List.foldr (fun p r => HAdd.hAdd (HMul.hMul (HPow.hPow b p.1) p.2) r) 0  …
    -/
    rw [CNF_ne_zero ho, foldr_cons, IH, div_add_mod]
    /-
      🎉 no goals
    -/


/-- Every exponent in the Cantor normal form `CNF b o` is less or equal to `log b o`. -/
theorem CNF_fst_le_log {b o : Ordinal.{u}} {x : Ordinal × Ordinal} :
    x ∈ CNF b o → x.1 ≤ log b o := by
  /-
    b o : Ordinal.{u}
    x : Prod Ordinal.{u} Ordinal.{u}
    ⊢ Membership.mem (Ordinal.CNF b o) x → LE.le x.1 (Ordinal.log b o)
  -/
  refine CNFRec b ?_ (fun o ho H ↦ ?_) o
    /-
      case refine_1
      b o : Ordinal.{u}
      x : Prod Ordinal.{u} Ordinal.{u}
      ⊢ Membership.mem (Ordinal.CNF b 0) x → LE.le x.1 (Ordinal.log b 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      b o✝ : Ordinal.{u}
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      H : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
      ⊢ Membership.mem (Ordinal.CNF b o) x → LE.le x.1 (Ordinal.log b o)
    -/
  · rw [CNF_ne_zero ho, mem_cons]
    /-
      case refine_2
      b o✝ : Ordinal.{u}
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      H : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
      ⊢ Or (Eq x { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal. …
    -/
    rintro (rfl | h)
      /-
        case refine_2.inl
        b o✝ o : Ordinal.{u}
        ho : Ne o 0
        H : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
        ⊢ LE.le { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.log …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        b o✝ : Ordinal.{u}
        x : Prod Ordinal.{u} Ordinal.{u}
        o : Ordinal.{u}
        ho : Ne o 0
        H : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
        h : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
        ⊢ LE.le x.1 (Ordinal.log b o)
      -/
    · exact (H h).trans (log_mono_right _ (mod_opow_log_lt_self b ho).le)
      /-
        🎉 no goals
      -/


/-- Every exponent in the Cantor normal form `CNF b o` is less or equal to `o`. -/
@[deprecated CNF_fst_le_log (since := "2024-09-21")]
theorem CNF_fst_le {b o : Ordinal.{u}} {x : Ordinal × Ordinal} (h : x ∈ CNF b o) : x.1 ≤ o :=
  (CNF_fst_le_log h).trans <| log_le_self _ _


/-- Every coefficient in a Cantor normal form is positive. -/
theorem CNF_lt_snd {b o : Ordinal.{u}} {x : Ordinal × Ordinal} : x ∈ CNF b o → 0 < x.2 := by
  /-
    b o : Ordinal.{u}
    x : Prod Ordinal.{u} Ordinal.{u}
    ⊢ Membership.mem (Ordinal.CNF b o) x → LT.lt 0 x.2
  -/
  refine CNFRec b (by simp) (fun o ho IH ↦ ?_) o
  /-
    b o✝ : Ordinal.{u}
    x : Prod Ordinal.{u} Ordinal.{u}
    o : Ordinal.{u}
    ho : Ne o 0
    IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
    ⊢ Membership.mem (Ordinal.CNF b o) x → LT.lt 0 x.2
  -/
  rw [CNF_ne_zero ho]
  /-
    b o✝ : Ordinal.{u}
    x : Prod Ordinal.{u} Ordinal.{u}
    o : Ordinal.{u}
    ho : Ne o 0
    IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
    ⊢ Membership.mem (List.cons { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow …
  -/
  rintro (h | ⟨_, h⟩)
    /-
      case head
      b o✝ o : Ordinal.{u}
      ho : Ne o 0
      IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
      ⊢ LT.lt 0 { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.l …
    -/
  · exact div_opow_log_pos b ho
    /-
      🎉 no goals
    -/
    /-
      case tail
      b o✝ : Ordinal.{u}
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
      h : List.Mem x (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o))))
      ⊢ LT.lt 0 x.2
    -/
  · exact IH h
    /-
      🎉 no goals
    -/


/-- Every coefficient in the Cantor normal form `CNF b o` is less than `b`. -/
theorem CNF_snd_lt {b o : Ordinal.{u}} (hb : 1 < b) {x : Ordinal × Ordinal} :
    x ∈ CNF b o → x.2 < b := by
  /-
    b o : Ordinal.{u}
    hb : LT.lt 1 b
    x : Prod Ordinal.{u} Ordinal.{u}
    ⊢ Membership.mem (Ordinal.CNF b o) x → LT.lt x.2 b
  -/
  refine CNFRec b ?_ (fun o ho IH ↦ ?_) o
    /-
      case refine_1
      b o : Ordinal.{u}
      hb : LT.lt 1 b
      x : Prod Ordinal.{u} Ordinal.{u}
      ⊢ Membership.mem (Ordinal.CNF b 0) x → LT.lt x.2 b
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      b o✝ : Ordinal.{u}
      hb : LT.lt 1 b
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
      ⊢ Membership.mem (Ordinal.CNF b o) x → LT.lt x.2 b
    -/
  · rw [CNF_ne_zero ho]
    /-
      case refine_2
      b o✝ : Ordinal.{u}
      hb : LT.lt 1 b
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
      ⊢ Membership.mem (List.cons { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow …
    -/
    intro h
    /-
      case refine_2
      b o✝ : Ordinal.{u}
      hb : LT.lt 1 b
      x : Prod Ordinal.{u} Ordinal.{u}
      o : Ordinal.{u}
      ho : Ne o 0
      IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
      h : Membership.mem (List.cons { fst := Ordinal.log b o, snd := HDiv.hDiv o (HP …
      ⊢ LT.lt x.2 b
    -/
    obtain rfl | h := mem_cons.mp h
      /-
        case refine_2.inl
        b o✝ : Ordinal.{u}
        hb : LT.lt 1 b
        o : Ordinal.{u}
        ho : Ne o 0
        IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
        h : Membership.mem (List.cons { fst := Ordinal.log b o, snd := HDiv.hDiv o (HP …
        ⊢ LT.lt { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.log …
      -/
    · exact div_opow_log_lt o hb
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        b o✝ : Ordinal.{u}
        hb : LT.lt 1 b
        x : Prod Ordinal.{u} Ordinal.{u}
        o : Ordinal.{u}
        ho : Ne o 0
        IH : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o) …
        h✝ : Membership.mem (List.cons { fst := Ordinal.log b o, snd := HDiv.hDiv o (H …
        h : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
        ⊢ LT.lt x.2 b
      -/
    · exact IH h
      /-
        🎉 no goals
      -/


/-- The exponents of the Cantor normal form are decreasing. -/
theorem CNF_sorted (b o : Ordinal) : ((CNF b o).map Prod.fst).Sorted (· > ·) := by
  /-
    b o : Ordinal.{u_1}
    ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
  -/
  refine CNFRec b ?_ (fun o ho IH ↦ ?_) o
    /-
      case refine_1
      b o : Ordinal.{u_1}
      ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b 0))
    -/
  · rw [CNF_zero]
    /-
      case refine_1
      b o : Ordinal.{u_1}
      ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst List.nil)
    -/
    exact sorted_nil
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      b o✝ o : Ordinal.{u_1}
      ho : Ne o 0
      IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
      ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
    -/
  · rcases le_or_lt b 1 with hb | hb
      /-
        case refine_2.inl
        b o✝ o : Ordinal.{u_1}
        ho : Ne o 0
        IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
        hb : LE.le b 1
        ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
      -/
    · rw [CNF_of_le_one hb ho]
      /-
        case refine_2.inl
        b o✝ o : Ordinal.{u_1}
        ho : Ne o 0
        IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
        hb : LE.le b 1
        ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (List.cons { fst : …
      -/
      exact sorted_singleton _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        b o✝ o : Ordinal.{u_1}
        ho : Ne o 0
        IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
        hb : LT.lt 1 b
        ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
      -/
    · obtain hob | hbo := lt_or_le o b
        /-
          case refine_2.inr.inl
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hob : LT.lt o b
          ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
        -/
      · rw [CNF_of_lt ho hob]
        /-
          case refine_2.inr.inl
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hob : LT.lt o b
          ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (List.cons { fst : …
        -/
        exact sorted_singleton _
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr.inr
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hbo : LE.le b o
          ⊢ List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b o))
        -/
      · rw [CNF_ne_zero ho, map_cons, sorted_cons]
        /-
          case refine_2.inr.inr
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hbo : LE.le b o
          ⊢ And (∀ (b_1 : Ordinal.{u_1}), Membership.mem (List.map Prod.fst (Ordinal.CNF …
        -/
        refine ⟨fun a H ↦ ?_, IH⟩
        /-
          case refine_2.inr.inr
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hbo : LE.le b o
          a : Ordinal.{u_1}
          H : Membership.mem (List.map Prod.fst (Ordinal.CNF b (HMod.hMod o (HPow.hPow b …
          ⊢ GT.gt { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.log …
        -/
        rw [mem_map] at H
        /-
          case refine_2.inr.inr
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hbo : LE.le b o
          a : Ordinal.{u_1}
          H : Exists fun a_1 => And (Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hP …
          ⊢ GT.gt { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.log …
        -/
        rcases H with ⟨⟨a, a'⟩, H, rfl⟩
        /-
          case refine_2.inr.inr.intro.mk.intro
          b o✝ o : Ordinal.{u_1}
          ho : Ne o 0
          IH : List.Sorted (fun x1 x2 => GT.gt x1 x2) (List.map Prod.fst (Ordinal.CNF b  …
          hb : LT.lt 1 b
          hbo : LE.le b o
          a a' : Ordinal.{u_1}
          H : Membership.mem (Ordinal.CNF b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)) …
          ⊢ GT.gt { fst := Ordinal.log b o, snd := HDiv.hDiv o (HPow.hPow b (Ordinal.log …
        -/
        exact (CNF_fst_le_log H).trans_lt (log_mod_opow_log_lt_log_self hb hbo)
        /-
          🎉 no goals
        -/


